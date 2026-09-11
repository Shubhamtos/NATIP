"""Sector-to-index mapping for dashboard sector charts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SectorIndex:
    """Yahoo Finance sector-index lookup result."""

    name: str
    yahoo_symbols: tuple[str, ...]


_SECTOR_INDEXES: dict[str, SectorIndex] = {
    "automobile": SectorIndex("Nifty Auto", ("^CNXAUTO",)),
    "auto": SectorIndex("Nifty Auto", ("^CNXAUTO",)),
    "basic materials": SectorIndex("Nifty Metal", ("^CNXMETAL",)),
    "capital goods": SectorIndex("Nifty Capital Goods", ("^CNXCAPITALGOODS", "^CNXINFRA")),
    "cement": SectorIndex("Nifty Cement", ("^CNXCEMENT", "^CNXINFRA")),
    "chemicals": SectorIndex("Nifty Chemicals", ("^CNXCHEMICALS",)),
    "communication services": SectorIndex("Nifty Telecommunications", ("^CNXIT", "^NSEI")),
    "consumer cyclical": SectorIndex("Nifty Auto", ("^CNXAUTO", "^CNXCONSUMPTION")),
    "consumer defensive": SectorIndex("Nifty FMCG", ("^CNXFMCG",)),
    "consumer durables": SectorIndex("Nifty Consumer Durables", ("^CNXCONSUMERDURABLES",)),
    "consumer goods": SectorIndex("Nifty FMCG", ("^CNXFMCG",)),
    "consumer non-durables": SectorIndex("Nifty FMCG", ("^CNXFMCG",)),
    "consumer services": SectorIndex("Nifty Consumer Services", ("^CNXCONSUMPTION", "^NSEI")),
    "diversified": SectorIndex("Nifty 50", ("^NSEI",)),
    "energy": SectorIndex("Nifty Energy", ("^CNXENERGY",)),
    "finance": SectorIndex("Nifty Financial Services", ("NIFTY_FIN_SERVICE.NS", "^NSEBANK", "^CNXFINANCE")),
    "financial": SectorIndex("Nifty Financial Services", ("NIFTY_FIN_SERVICE.NS", "^NSEBANK", "^CNXFINANCE")),
    "financial services": SectorIndex("Nifty Financial Services", ("NIFTY_FIN_SERVICE.NS", "^NSEBANK", "^CNXFINANCE")),
    "financials": SectorIndex("Nifty Financial Services", ("NIFTY_FIN_SERVICE.NS", "^NSEBANK", "^CNXFINANCE")),
    "healthcare": SectorIndex("Nifty Healthcare/Pharma", ("^CNXPHARMA",)),
    "industrials": SectorIndex("Nifty Capital Goods", ("^CNXCAPITALGOODS", "^CNXINFRA")),
    "infrastructure": SectorIndex("Nifty Infrastructure", ("^CNXINFRA",)),
    "information technology": SectorIndex("Nifty IT", ("^CNXIT",)),
    "materials": SectorIndex("Nifty Metal", ("^CNXMETAL",)),
    "metals": SectorIndex("Nifty Metal", ("^CNXMETAL",)),
    "oil gas & consumable fuels": SectorIndex("Nifty Energy", ("^CNXENERGY",)),
    "oil gas and consumable fuels": SectorIndex("Nifty Energy", ("^CNXENERGY",)),
    "oil and gas": SectorIndex("Nifty Oil & Gas", ("^CNXENERGY",)),
    "pharma": SectorIndex("Nifty Pharma", ("^CNXPHARMA",)),
    "power": SectorIndex("Nifty Energy", ("^CNXENERGY",)),
    "real estate": SectorIndex("Nifty Realty", ("^CNXREALTY",)),
    "realty": SectorIndex("Nifty Realty", ("^CNXREALTY",)),
    "technology": SectorIndex("Nifty IT", ("^CNXIT",)),
    "telecom": SectorIndex("Nifty Telecommunications", ("^CNXIT", "^NSEI")),
    "utilities": SectorIndex("Nifty Energy", ("^CNXENERGY",)),
}


def sector_index_for_sector(sector: str | None) -> SectorIndex | None:
    """Return the best matching sector index for a sector label.

    Args:
        sector: Company sector from Yahoo Finance or local NSE catalog.

    Returns:
        Sector index metadata when a free Yahoo Finance proxy is known.
    """

    if not sector:
        return None
    return _SECTOR_INDEXES.get(sector.strip().lower())
