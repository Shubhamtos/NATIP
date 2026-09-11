"""Nifty 250 universe loading for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.dashboard.nse_symbols import NSE_SYMBOLS, NseSymbol

NIFTY_ARCHIVE_BASE_URL = "https://nsearchives.nseindia.com/content/indices"
NIFTY_100_CSV_URL = f"{NIFTY_ARCHIVE_BASE_URL}/ind_nifty100list.csv"
NIFTY_50_CSV_URL = f"{NIFTY_ARCHIVE_BASE_URL}/ind_nifty50list.csv"
NIFTY_MIDCAP_150_CSV_URL = f"{NIFTY_ARCHIVE_BASE_URL}/ind_niftymidcap150list.csv"
NIFTY_SMALLCAP_250_CSV_URL = f"{NIFTY_ARCHIVE_BASE_URL}/ind_niftysmallcap250list.csv"
NIFTY_MICROCAP_250_CSV_URL = f"{NIFTY_ARCHIVE_BASE_URL}/ind_niftymicrocap250_list.csv"
NIFTY_CSV_STORAGE_OPTIONS = {"User-Agent": "Mozilla/5.0"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NIFTY_SMALLCAP_250_CSV = PROJECT_ROOT / "data" / "nifty_smallcap250_constituents.csv"

_NIFTY_NEXT_50_SYMBOLS = (
    "ABB",
    "ADANIENSOL",
    "ADANIGREEN",
    "ADANIPOWER",
    "AMBUJACEM",
    "BAJAJHLDNG",
    "BANKBARODA",
    "BHEL",
    "BOSCHLTD",
    "CANBK",
    "ABCAPITAL",
    "CGPOWER",
    "CHOLAFIN",
    "DABUR",
    "DLF",
    "DMART",
    "GAIL",
    "GODREJCP",
    "HAL",
    "HAVELLS",
    "INDHOTEL",
    "ICICIGI",
    "ICICIPRULI",
    "INDIGO",
    "IRFC",
    "JINDALSTEL",
    "JSWENERGY",
    "LICI",
    "LODHA",
    "LTIM",
    "MANKIND",
    "MAXHEALTH",
    "MOTHERSON",
    "NAUKRI",
    "NHPC",
    "PFC",
    "PIDILITIND",
    "PNB",
    "RECLTD",
    "SHREECEM",
    "SIEMENS",
    "TATAPOWER",
    "TORNTPHARM",
    "TRENT",
    "TVSMOTOR",
    "UNIONBANK",
    "VBL",
    "VEDL",
    "ZOMATO",
    "ZYDUSLIFE",
)

_NIFTY_MIDCAP_150_SYMBOLS = (
    "ABCAPITAL",
    "ACC",
    "APLAPOLLO",
    "AUBANK",
    "ALKEM",
    "ASHOKLEY",
    "ASTRAL",
    "AUROPHARMA",
    "BALKRISIND",
    "BANDHANBNK",
    "BANKINDIA",
    "BDL",
    "BEL",
    "BHARATFORG",
    "BIOCON",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "CUMMINSIND",
    "DALBHARAT",
    "DEEPAKNTR",
    "DELHIVERY",
    "DIXON",
    "ESCORTS",
    "FEDERALBNK",
    "FORTIS",
    "GLAND",
    "GMRAIRPORT",
    "GODREJPROP",
    "GUJGASLTD",
    "HDFCAMC",
    "HINDPETRO",
    "HONAUT",
    "IDFCFIRSTB",
    "IGL",
    "INDHOTEL",
    "INDUSTOWER",
    "IPCALAB",
    "JSL",
    "JSWSTEEL",
    "JUBLFOOD",
    "LALPATHLAB",
    "LAURUSLABS",
    "LICHSGFIN",
    "LTTS",
    "LUPIN",
    "M&MFIN",
    "MANAPPURAM",
    "MARICO",
    "MFSL",
    "MPHASIS",
    "MRF",
    "NMDC",
    "OBEROIRLTY",
    "OFSS",
    "PAGEIND",
    "PAYTM",
    "PERSISTENT",
    "PETRONET",
    "PHOENIXLTD",
    "POLICYBZR",
    "POLYCAB",
    "PRESTIGE",
    "RAMCOCEM",
    "SAIL",
    "SONACOMS",
    "SRF",
    "SUNTV",
    "SUPREMEIND",
    "SYNGENE",
    "TATACHEM",
    "TATACOMM",
    "TATAELXSI",
    "TATATECH",
    "TIINDIA",
    "TORNTPOWER",
    "UPL",
    "YESBANK",
    "ZEEL",
    "360ONE",
    "AARTIIND",
    "ABFRL",
    "ATGL",
    "AJANTPHARM",
    "APARINDS",
    "APOLLOTYRE",
    "BATAINDIA",
    "BAYERCROP",
    "BERGEPAINT",
    "BLUESTARCO",
    "BRIGADE",
    "CARBORUNIV",
    "CASTROLIND",
    "CROMPTON",
    "CREDITACC",
    "ENDURANCE",
    "EXIDEIND",
    "FACT",
    "FLUOROCHEM",
    "GLAXO",
    "GLENMARK",
    "GODREJIND",
    "GRANULES",
    "HINDCOPPER",
    "HUDCO",
    "IEX",
    "INDIANB",
    "IRB",
    "JBCHEPHARM",
    "JINDALSAW",
    "KALYANKJIL",
    "KEI",
    "KPITTECH",
    "LINDEINDIA",
    "MAHABANK",
    "MAZDOCK",
    "MOTILALOFS",
    "NATCOPHARM",
    "NATIONALUM",
    "NHPC",
    "NLCINDIA",
    "PATANJALI",
    "PEL",
    "POONAWALLA",
    "RAILTEL",
    "RBLBANK",
    "RVNL",
    "SCHAEFFLER",
    "SJVN",
    "SOLARINDS",
    "STARHEALTH",
    "SUNDARMFIN",
    "SUZLON",
    "TATAINVEST",
    "THERMAX",
    "UNOMINDA",
    "VOLTAS",
    "WHIRLPOOL",
    "ZFCVINDIA",
    "CENTRALBK",
    "COCHINSHIP",
    "KIMS",
    "KPRMILL",
    "MUTHOOTFIN",
    "OIL",
    "OLECTRA",
    "POWERINDIA",
    "REDINGTON",
    "SONATSOFTW",
    "TRIDENT",
)


@dataclass(frozen=True, slots=True)
class UniverseLoadResult:
    """Loaded stock universe."""

    symbols: tuple[NseSymbol, ...]
    source: str
    warning: str | None = None


def fast_nifty50_universe() -> UniverseLoadResult:
    """Return a fast local Nifty 50 universe for UI rendering."""

    return UniverseLoadResult(symbols=NSE_SYMBOLS, source="fast local Nifty 50 catalog")


def fast_nifty100_universe() -> UniverseLoadResult:
    """Return a fast local Nifty 100 universe for UI rendering."""

    return UniverseLoadResult(
        symbols=_symbols_from_catalog(_NIFTY_NEXT_50_SYMBOLS, base_symbols=NSE_SYMBOLS),
        source="fast local Nifty 100 catalog",
    )


def fast_nifty250_universe() -> UniverseLoadResult:
    """Return a fast local Nifty 250 universe for UI/model scans."""

    records_by_symbol: dict[str, NseSymbol] = {}
    for record in fast_nifty100_universe().symbols:
        records_by_symbol[record.symbol] = record
    for record in fast_nifty_midcap150_universe().symbols:
        records_by_symbol[record.symbol] = record
    return UniverseLoadResult(
        symbols=tuple(records_by_symbol.values()),
        source="fast local Nifty 100 + Nifty Midcap 150 catalog",
    )


def fast_nifty_midcap150_universe() -> UniverseLoadResult:
    """Return a fast local Nifty Midcap 150 universe for UI rendering."""

    return UniverseLoadResult(
        symbols=_symbols_from_catalog(_NIFTY_MIDCAP_150_SYMBOLS),
        source="fast local Nifty Midcap 150 catalog",
    )


def fast_nifty_smallcap250_universe() -> UniverseLoadResult:
    """Return a local Nifty Smallcap 250 snapshot for UI/model scans."""

    records = _records_from_frame(pd.read_csv(LOCAL_NIFTY_SMALLCAP_250_CSV))
    return UniverseLoadResult(symbols=tuple(records), source=str(LOCAL_NIFTY_SMALLCAP_250_CSV))


def load_nifty250_universe() -> UniverseLoadResult:
    """Load the current Nifty 250 universe.

    Returns:
        Nifty 100 plus Nifty Midcap 150 from Nifty Indices, or a curated
        fallback when the source cannot be reached.
    """

    try:
        frames = [_read_nifty_csv(NIFTY_100_CSV_URL), _read_nifty_csv(NIFTY_MIDCAP_150_CSV_URL)]
        records_by_symbol: dict[str, NseSymbol] = {}
        for frame in frames:
            for record in _records_from_frame(frame):
                records_by_symbol[record.symbol] = record
        records = list(records_by_symbol.values())
        if records:
            return UniverseLoadResult(
                symbols=tuple(records),
                source=f"{NIFTY_100_CSV_URL} + {NIFTY_MIDCAP_150_CSV_URL}",
            )
    except Exception as exc:
        return UniverseLoadResult(
            symbols=NSE_SYMBOLS,
            source="curated fallback",
            warning=f"Could not load official Nifty 250 CSV sources: {exc}",
        )

    return UniverseLoadResult(
        symbols=NSE_SYMBOLS,
        source="curated fallback",
        warning="Official Nifty 250 CSV sources returned no symbols.",
    )


def load_nifty100_universe() -> UniverseLoadResult:
    """Load the current Nifty 100 universe."""

    return _load_single_csv_universe(
        url=NIFTY_100_CSV_URL,
        fallback_warning="Could not load official Nifty 100 CSV",
        empty_warning="Official Nifty 100 CSV returned no symbols.",
    )


def load_nifty_midcap150_universe() -> UniverseLoadResult:
    """Load the current Nifty Midcap 150 universe."""

    return _load_single_csv_universe(
        url=NIFTY_MIDCAP_150_CSV_URL,
        fallback_warning="Could not load official Nifty Midcap 150 CSV",
        empty_warning="Official Nifty Midcap 150 CSV returned no symbols.",
    )


def load_nifty50_universe() -> UniverseLoadResult:
    """Load the current Nifty 50 universe."""

    return _load_single_csv_universe(
        url=NIFTY_50_CSV_URL,
        fallback_warning="Could not load official Nifty 50 CSV",
        empty_warning="Official Nifty 50 CSV returned no symbols.",
    )


def load_nifty_smallcap250_universe() -> UniverseLoadResult:
    """Load the current Nifty Smallcap 250 universe."""

    try:
        records = _records_from_frame(_read_nifty_csv(NIFTY_SMALLCAP_250_CSV_URL))
        if records:
            return UniverseLoadResult(symbols=tuple(records), source=NIFTY_SMALLCAP_250_CSV_URL)
        warning = "Official Nifty Smallcap 250 CSV returned no symbols."
    except Exception as exc:
        warning = f"Could not load official Nifty Smallcap 250 CSV: {exc}"

    fallback = fast_nifty_smallcap250_universe()
    return UniverseLoadResult(
        symbols=fallback.symbols,
        source=fallback.source,
        warning=f"{warning} Using local official snapshot.",
    )


def load_nifty_microcap250_universe() -> UniverseLoadResult:
    """Load the current Nifty Microcap 250 universe."""

    return _load_single_csv_universe(
        url=NIFTY_MICROCAP_250_CSV_URL,
        fallback_warning="Could not load official Nifty Microcap 250 CSV",
        empty_warning="Official Nifty Microcap 250 CSV returned no symbols.",
        fallback_symbols=(),
    )


def universe_labels(symbols: tuple[NseSymbol, ...]) -> list[str]:
    """Return display labels for a universe."""

    return [item.label for item in symbols]


def _load_single_csv_universe(
    *,
    url: str,
    fallback_warning: str,
    empty_warning: str,
    fallback_symbols: tuple[NseSymbol, ...] = NSE_SYMBOLS,
) -> UniverseLoadResult:
    """Load one official Nifty Indices CSV universe."""

    try:
        records = _records_from_frame(_read_nifty_csv(url))
        if records:
            return UniverseLoadResult(symbols=tuple(records), source=url)
    except Exception as exc:
        return UniverseLoadResult(
            symbols=fallback_symbols,
            source="curated fallback",
            warning=f"{fallback_warning}: {exc}",
        )

    return UniverseLoadResult(
        symbols=fallback_symbols,
        source="curated fallback",
        warning=empty_warning,
    )


def _read_nifty_csv(url: str) -> pd.DataFrame:
    """Read an official NSE index constituent CSV."""

    return pd.read_csv(url, storage_options=NIFTY_CSV_STORAGE_OPTIONS)


def _records_from_frame(frame: pd.DataFrame) -> list[NseSymbol]:
    """Build symbol records from an official Nifty Indices frame."""

    return [
        NseSymbol(
            symbol=str(row["Symbol"]).strip().upper(),
            name=str(row["Company Name"]).strip(),
            sector=str(row.get("Industry", "Unknown")).strip() or "Unknown",
        )
        for _, row in frame.iterrows()
        if str(row.get("Symbol", "")).strip()
    ]


def _symbols_from_catalog(
    symbols: tuple[str, ...],
    *,
    base_symbols: tuple[NseSymbol, ...] = (),
) -> tuple[NseSymbol, ...]:
    """Build a de-duplicated local universe from known and symbol-only records."""

    records_by_symbol = {record.symbol: record for record in base_symbols}
    for symbol in symbols:
        records_by_symbol.setdefault(
            symbol,
            NseSymbol(symbol=symbol, name=symbol, sector="Unknown"),
        )

    return tuple(records_by_symbol.values())
