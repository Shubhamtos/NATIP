"""SQLite/SQLAlchemy storage for raw-material impact research."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import PROJECT_ROOT
from app.intelligence.raw_material.models import (
    CompanyRawMaterialMapping,
    RawMaterialImpactSignal,
    RawMaterialMaster,
    RawMaterialPricePoint,
    RelationshipType,
    VerificationStatus,
)


RAW_MATERIAL_DB_PATH = PROJECT_ROOT / "data" / "raw_material" / "raw_material_impact.db"
RAW_MATERIAL_MASTER_CSV = PROJECT_ROOT / "data" / "raw_material" / "raw_material_master.csv"
COMPANY_MAPPING_CSV = PROJECT_ROOT / "data" / "raw_material" / "company_material_mappings.csv"


class Base(DeclarativeBase):
    """Raw-material SQLAlchemy base."""


class RawMaterialMasterRow(Base):
    """Tracked raw material."""

    __tablename__ = "raw_material_master"

    raw_material_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[str] = mapped_column(String(120), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(160), nullable=False)
    yahoo_symbol: Mapped[str | None] = mapped_column(String(80))
    original_currency: Mapped[str] = mapped_column(String(12), nullable=False)
    unit: Mapped[str] = mapped_column(String(40), nullable=False)
    is_international: Mapped[bool] = mapped_column(Boolean, default=True)
    related_sectors_json: Mapped[str] = mapped_column(Text, default="[]")
    source: Mapped[str] = mapped_column(String(160), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class RawMaterialPriceHistoryRow(Base):
    """Raw-material price history."""

    __tablename__ = "raw_material_price_history"
    __table_args__ = (
        UniqueConstraint("raw_material_id", "timestamp", "source", name="uq_raw_material_price"),
        Index("ix_raw_material_price_material_time", "raw_material_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_material_id: Mapped[str] = mapped_column(ForeignKey("raw_material_master.raw_material_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(12), nullable=False)
    unit: Mapped[str] = mapped_column(String(40), nullable=False)
    retrieval_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(160), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    data_status: Mapped[str] = mapped_column(String(60), default="OK")
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)


class CompanyRawMaterialMappingRow(Base):
    """Versioned company exposure mapping."""

    __tablename__ = "company_raw_material_mapping"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "raw_material_id",
            "effective_from",
            name="uq_company_raw_material_mapping_version",
        ),
        Index("ix_company_material_symbol", "symbol"),
        Index("ix_company_material_material", "raw_material_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    company_name: Mapped[str] = mapped_column(String(180), nullable=False)
    sector: Mapped[str] = mapped_column(String(120), nullable=False)
    raw_material_id: Mapped[str] = mapped_column(ForeignKey("raw_material_master.raw_material_id"))
    raw_material_name: Mapped[str] = mapped_column(String(160), nullable=False)
    relationship: Mapped[str] = mapped_column(String(40), nullable=False)
    material_cost_pct_cogs: Mapped[float | None] = mapped_column(Float)
    material_spend_pct_revenue: Mapped[float | None] = mapped_column(Float)
    import_dependency: Mapped[float | None] = mapped_column(Float)
    source_countries_json: Mapped[str] = mapped_column(Text, default="[]")
    supplier_concentration: Mapped[float | None] = mapped_column(Float)
    hedge_ratio: Mapped[float | None] = mapped_column(Float)
    hedge_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    inventory_days: Mapped[float | None] = mapped_column(Float)
    pass_through_ratio: Mapped[float | None] = mapped_column(Float)
    pass_through_lag_days: Mapped[int | None] = mapped_column(Integer)
    pricing_power: Mapped[str | None] = mapped_column(String(80))
    vertical_integration: Mapped[str | None] = mapped_column(String(120))
    historical_sensitivity: Mapped[float | None] = mapped_column(Float)
    evidence_source_url: Mapped[str | None] = mapped_column(Text)
    evidence_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verification_status: Mapped[str] = mapped_column(String(60), default=VerificationStatus.TEMPLATE_UNVERIFIED.value)
    analyst_notes: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class RawMaterialImpactSignalRow(Base):
    """Stored impact signal."""

    __tablename__ = "raw_material_impact_signal"
    __table_args__ = (
        UniqueConstraint("symbol", "raw_material_id", "as_of", name="uq_raw_material_signal"),
        Index("ix_raw_material_signal_symbol_time", "symbol", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    raw_material_id: Mapped[str] = mapped_column(String(80), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    alert_level: Mapped[str] = mapped_column(String(60), nullable=False)
    alert_priority: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class RawMaterialImpactStore:
    """Local-first raw-material impact store."""

    def __init__(self, db_path: Path = RAW_MATERIAL_DB_PATH) -> None:
        """Initialize the store."""

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def initialize(self) -> None:
        """Create database tables."""

        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterable[Session]:
        """Yield a managed session."""

        with self.session_factory() as session:
            yield session

    def seed_from_csv(self) -> None:
        """Seed raw-material master and mapping templates when available."""

        self.initialize()
        if RAW_MATERIAL_MASTER_CSV.exists():
            frame = pd.read_csv(RAW_MATERIAL_MASTER_CSV).fillna("")
            for _, row in frame.iterrows():
                self.upsert_material(_material_from_csv_row(row.to_dict()))
        if COMPANY_MAPPING_CSV.exists():
            frame = pd.read_csv(COMPANY_MAPPING_CSV).fillna("")
            for _, row in frame.iterrows():
                self.upsert_mapping(_mapping_from_csv_row(row.to_dict()))

    def upsert_material(self, material: RawMaterialMaster) -> None:
        """Insert or update a raw material."""

        with self.session() as session:
            existing = session.get(RawMaterialMasterRow, material.raw_material_id)
            values = {
                "name": material.name,
                "category": material.category,
                "benchmark": material.benchmark,
                "yahoo_symbol": material.yahoo_symbol,
                "original_currency": material.original_currency,
                "unit": material.unit,
                "is_international": material.is_international,
                "related_sectors_json": json.dumps(material.related_sectors),
                "source": material.source,
                "source_url": material.source_url,
                "updated_at": datetime.now(UTC),
            }
            if existing is None:
                session.add(RawMaterialMasterRow(raw_material_id=material.raw_material_id, **values))
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
            session.commit()

    def upsert_mapping(self, mapping: CompanyRawMaterialMapping) -> None:
        """Insert or update a versioned mapping."""

        with self.session() as session:
            existing = session.scalar(
                select(CompanyRawMaterialMappingRow).where(
                    CompanyRawMaterialMappingRow.symbol == mapping.symbol,
                    CompanyRawMaterialMappingRow.raw_material_id == mapping.raw_material_id,
                    CompanyRawMaterialMappingRow.effective_from == mapping.effective_from,
                )
            )
            values = _mapping_to_row_values(mapping)
            if existing is None:
                session.add(CompanyRawMaterialMappingRow(**values))
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
                existing.updated_at = datetime.now(UTC)
            session.commit()

    def add_price_points(self, points: list[RawMaterialPricePoint]) -> int:
        """Add price points idempotently."""

        inserted = 0
        with self.session() as session:
            for point in points:
                exists = session.scalar(
                    select(RawMaterialPriceHistoryRow.id).where(
                        RawMaterialPriceHistoryRow.raw_material_id == point.raw_material_id,
                        RawMaterialPriceHistoryRow.timestamp == point.timestamp,
                        RawMaterialPriceHistoryRow.source == point.source,
                    )
                )
                if exists is not None:
                    continue
                session.add(RawMaterialPriceHistoryRow(**point.model_dump()))
                inserted += 1
            session.commit()
        return inserted

    def list_materials(self) -> list[RawMaterialMaster]:
        """Return tracked materials."""

        self.seed_from_csv()
        with self.session() as session:
            rows = session.scalars(select(RawMaterialMasterRow).order_by(RawMaterialMasterRow.name)).all()
        return [_material_from_row(row) for row in rows]

    def list_mappings(self, *, symbol: str | None = None, active_only: bool = True) -> list[CompanyRawMaterialMapping]:
        """Return company-material mappings."""

        self.seed_from_csv()
        statement = select(CompanyRawMaterialMappingRow)
        if symbol:
            statement = statement.where(CompanyRawMaterialMappingRow.symbol == symbol.strip().upper().replace(".NS", ""))
        if active_only:
            statement = statement.where(CompanyRawMaterialMappingRow.active.is_(True))
        statement = statement.order_by(CompanyRawMaterialMappingRow.symbol, CompanyRawMaterialMappingRow.raw_material_name)
        with self.session() as session:
            rows = session.scalars(statement).all()
        return [_mapping_from_row(row) for row in rows]

    def price_history(self, raw_material_id: str) -> pd.DataFrame:
        """Return raw-material price history as a DataFrame."""

        self.seed_from_csv()
        with self.session() as session:
            rows = session.scalars(
                select(RawMaterialPriceHistoryRow)
                .where(RawMaterialPriceHistoryRow.raw_material_id == raw_material_id)
                .order_by(RawMaterialPriceHistoryRow.timestamp)
            ).all()
        return pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp,
                    "price": row.price,
                    "currency": row.currency,
                    "unit": row.unit,
                    "source": row.source,
                    "quality_score": row.quality_score,
                }
                for row in rows
            ]
        )

    def store_signal(self, signal: RawMaterialImpactSignal) -> None:
        """Store an impact signal idempotently."""

        with self.session() as session:
            existing = session.scalar(
                select(RawMaterialImpactSignalRow).where(
                    RawMaterialImpactSignalRow.symbol == signal.symbol,
                    RawMaterialImpactSignalRow.raw_material_id == signal.raw_material_id,
                    RawMaterialImpactSignalRow.as_of == signal.as_of,
                )
            )
            payload = signal.model_dump_json()
            if existing is None:
                session.add(
                    RawMaterialImpactSignalRow(
                        symbol=signal.symbol,
                        raw_material_id=signal.raw_material_id,
                        as_of=signal.as_of,
                        payload_json=payload,
                        alert_level=signal.alert_level.value,
                        alert_priority=signal.alert_priority,
                    )
                )
            else:
                existing.payload_json = payload
                existing.alert_level = signal.alert_level.value
                existing.alert_priority = signal.alert_priority
            session.commit()


def _material_from_csv_row(row: dict[str, Any]) -> RawMaterialMaster:
    return RawMaterialMaster(
        raw_material_id=str(row["raw_material_id"]),
        name=str(row["name"]),
        category=str(row["category"]),
        benchmark=str(row["benchmark"]),
        yahoo_symbol=str(row.get("yahoo_symbol") or "") or None,
        original_currency=str(row["original_currency"]),
        unit=str(row["unit"]),
        is_international=str(row.get("is_international", "true")).lower() == "true",
        related_sectors=[item.strip() for item in str(row.get("related_sectors") or "").split(";") if item.strip()],
        source=str(row["source"]),
        source_url=str(row.get("source_url") or "") or None,
    )


def _mapping_from_csv_row(row: dict[str, Any]) -> CompanyRawMaterialMapping:
    return CompanyRawMaterialMapping(
        symbol=str(row["symbol"]).strip().upper().replace(".NS", ""),
        company_name=str(row["company_name"]),
        sector=str(row["sector"]),
        raw_material_id=str(row["raw_material_id"]),
        raw_material_name=str(row["raw_material_name"]),
        relationship=RelationshipType(str(row["relationship"])),
        material_cost_pct_cogs=_optional_float(row.get("material_cost_pct_cogs")),
        material_spend_pct_revenue=_optional_float(row.get("material_spend_pct_revenue")),
        import_dependency=_optional_float(row.get("import_dependency")),
        source_countries=[item.strip() for item in str(row.get("source_countries") or "").split(";") if item.strip()],
        supplier_concentration=_optional_float(row.get("supplier_concentration")),
        hedge_ratio=_optional_float(row.get("hedge_ratio")),
        hedge_expiry=_optional_datetime(row.get("hedge_expiry")),
        inventory_days=_optional_float(row.get("inventory_days")),
        pass_through_ratio=_optional_float(row.get("pass_through_ratio")),
        pass_through_lag_days=_optional_int(row.get("pass_through_lag_days")),
        pricing_power=str(row.get("pricing_power") or "") or None,
        vertical_integration=str(row.get("vertical_integration") or "") or None,
        historical_sensitivity=_optional_float(row.get("historical_sensitivity")),
        evidence_source_url=str(row.get("evidence_source_url") or "") or None,
        evidence_date=_optional_datetime(row.get("evidence_date")),
        verification_status=VerificationStatus(str(row.get("verification_status") or VerificationStatus.TEMPLATE_UNVERIFIED.value)),
        analyst_notes=str(row.get("analyst_notes") or "") or None,
        confidence=_optional_float(row.get("confidence")),
        effective_from=_optional_datetime(row.get("effective_from")),
        effective_to=_optional_datetime(row.get("effective_to")),
        active=str(row.get("active", "true")).lower() == "true",
    )


def _material_from_row(row: RawMaterialMasterRow) -> RawMaterialMaster:
    return RawMaterialMaster(
        raw_material_id=row.raw_material_id,
        name=row.name,
        category=row.category,
        benchmark=row.benchmark,
        yahoo_symbol=row.yahoo_symbol,
        original_currency=row.original_currency,
        unit=row.unit,
        is_international=row.is_international,
        related_sectors=json.loads(row.related_sectors_json or "[]"),
        source=row.source,
        source_url=row.source_url,
    )


def _mapping_from_row(row: CompanyRawMaterialMappingRow) -> CompanyRawMaterialMapping:
    return CompanyRawMaterialMapping(
        symbol=row.symbol,
        company_name=row.company_name,
        sector=row.sector,
        raw_material_id=row.raw_material_id,
        raw_material_name=row.raw_material_name,
        relationship=RelationshipType(row.relationship),
        material_cost_pct_cogs=row.material_cost_pct_cogs,
        material_spend_pct_revenue=row.material_spend_pct_revenue,
        import_dependency=row.import_dependency,
        source_countries=json.loads(row.source_countries_json or "[]"),
        supplier_concentration=row.supplier_concentration,
        hedge_ratio=row.hedge_ratio,
        hedge_expiry=row.hedge_expiry,
        inventory_days=row.inventory_days,
        pass_through_ratio=row.pass_through_ratio,
        pass_through_lag_days=row.pass_through_lag_days,
        pricing_power=row.pricing_power,
        vertical_integration=row.vertical_integration,
        historical_sensitivity=row.historical_sensitivity,
        evidence_source_url=row.evidence_source_url,
        evidence_date=row.evidence_date,
        verification_status=VerificationStatus(row.verification_status),
        analyst_notes=row.analyst_notes,
        confidence=row.confidence,
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        active=row.active,
    )


def _mapping_to_row_values(mapping: CompanyRawMaterialMapping) -> dict[str, Any]:
    values = mapping.model_dump()
    values["relationship"] = mapping.relationship.value
    values["source_countries_json"] = json.dumps(mapping.source_countries)
    values["verification_status"] = mapping.verification_status.value
    values["evidence_source_url"] = str(mapping.evidence_source_url) if mapping.evidence_source_url else None
    values.pop("source_countries", None)
    return values


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _optional_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    py_dt = timestamp.to_pydatetime()
    return py_dt if py_dt.tzinfo else py_dt.replace(tzinfo=UTC)
