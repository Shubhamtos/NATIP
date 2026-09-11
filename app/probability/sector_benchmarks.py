"""Sector benchmark mapping for probability features."""

from __future__ import annotations

import pandas as pd

SECTOR_YAHOO_SYMBOLS: dict[str, str] = {
    "Automobile": "^CNXAUTO",
    "Capital Goods": "^CNXINFRA",
    "Cement": "^CNXREALTY",
    "Consumer Durables": "^CNXCONSUM",
    "Consumer Goods": "^CNXFMCG",
    "Consumer Services": "^CNXCONSUM",
    "Energy": "^CNXENERGY",
    "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Healthcare": "^CNXPHARMA",
    "Information Technology": "^CNXIT",
    "Infrastructure": "^CNXINFRA",
    "Industrials": "^CNXINFRA",
    "Metals": "^CNXMETAL",
    "Power": "^CNXENERGY",
    "Real Estate": "^CNXREALTY",
    "Services": "^CNXSERVICE",
    "Telecom": "^CNXMEDIA",
}


def sector_benchmark_symbols(universe: pd.DataFrame) -> list[str]:
    """Return unique Yahoo sector benchmark symbols for a universe."""

    symbols = [
        SECTOR_YAHOO_SYMBOLS[sector]
        for sector in universe["Sector"].dropna().astype(str).unique()
        if sector in SECTOR_YAHOO_SYMBOLS
    ]
    return list(dict.fromkeys(symbols))
