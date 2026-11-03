"""Curated NSE symbol catalog for the dashboard."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NseSymbol:
    """NSE stock symbol option."""

    symbol: str
    name: str
    sector: str

    @property
    def label(self) -> str:
        """Return UI label.

        Returns:
            Human-readable symbol label.
        """

        return f"{self.symbol} - {self.name}"


NSE_SYMBOLS: tuple[NseSymbol, ...] = (
    NseSymbol("RELIANCE", "Reliance Industries", "Energy"),
    NseSymbol("TCS", "Tata Consultancy Services", "Information Technology"),
    NseSymbol("HDFCBANK", "HDFC Bank", "Financial Services"),
    NseSymbol("ICICIBANK", "ICICI Bank", "Financial Services"),
    NseSymbol("INFY", "Infosys", "Information Technology"),
    NseSymbol("BHARTIARTL", "Bharti Airtel", "Telecom"),
    NseSymbol("SBIN", "State Bank of India", "Financial Services"),
    NseSymbol("ITC", "ITC", "Consumer Goods"),
    NseSymbol("LT", "Larsen & Toubro", "Capital Goods"),
    NseSymbol("HINDUNILVR", "Hindustan Unilever", "Consumer Goods"),
    NseSymbol("KOTAKBANK", "Kotak Mahindra Bank", "Financial Services"),
    NseSymbol("AXISBANK", "Axis Bank", "Financial Services"),
    NseSymbol("BAJFINANCE", "Bajaj Finance", "Financial Services"),
    NseSymbol("ASIANPAINT", "Asian Paints", "Consumer Durables"),
    NseSymbol("MARUTI", "Maruti Suzuki India", "Automobile"),
    NseSymbol("SUNPHARMA", "Sun Pharmaceutical", "Healthcare"),
    NseSymbol("TITAN", "Titan Company", "Consumer Durables"),
    NseSymbol("ULTRACEMCO", "UltraTech Cement", "Cement"),
    NseSymbol("WIPRO", "Wipro", "Information Technology"),
    NseSymbol("ONGC", "Oil and Natural Gas Corporation", "Energy"),
    NseSymbol("NTPC", "NTPC", "Power"),
    NseSymbol("POWERGRID", "Power Grid Corporation", "Power"),
    NseSymbol("M&M", "Mahindra & Mahindra", "Automobile"),
    NseSymbol("TMPV", "Tata Motors Passenger Vehicles", "Automobile"),
    NseSymbol("HCLTECH", "HCL Technologies", "Information Technology"),
    NseSymbol("TECHM", "Tech Mahindra", "Information Technology"),
    NseSymbol("NESTLEIND", "Nestle India", "Consumer Goods"),
    NseSymbol("JSWSTEEL", "JSW Steel", "Metals"),
    NseSymbol("TATASTEEL", "Tata Steel", "Metals"),
    NseSymbol("HINDALCO", "Hindalco Industries", "Metals"),
    NseSymbol("COALINDIA", "Coal India", "Energy"),
    NseSymbol("ADANIENT", "Adani Enterprises", "Diversified"),
    NseSymbol("ADANIPORTS", "Adani Ports and SEZ", "Infrastructure"),
    NseSymbol("GRASIM", "Grasim Industries", "Diversified"),
    NseSymbol("CIPLA", "Cipla", "Healthcare"),
    NseSymbol("DRREDDY", "Dr. Reddy's Laboratories", "Healthcare"),
    NseSymbol("DIVISLAB", "Divi's Laboratories", "Healthcare"),
    NseSymbol("APOLLOHOSP", "Apollo Hospitals", "Healthcare"),
    NseSymbol("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    NseSymbol("HDFCLIFE", "HDFC Life Insurance", "Financial Services"),
    NseSymbol("SBILIFE", "SBI Life Insurance", "Financial Services"),
    NseSymbol("HEROMOTOCO", "Hero MotoCorp", "Automobile"),
    NseSymbol("EICHERMOT", "Eicher Motors", "Automobile"),
    NseSymbol("BRITANNIA", "Britannia Industries", "Consumer Goods"),
    NseSymbol("TATACONSUM", "Tata Consumer Products", "Consumer Goods"),
    NseSymbol("UPL", "UPL", "Chemicals"),
    NseSymbol("BPCL", "Bharat Petroleum", "Energy"),
    NseSymbol("IOC", "Indian Oil Corporation", "Energy"),
    NseSymbol("INDUSINDBK", "IndusInd Bank", "Financial Services"),
    NseSymbol("BAJAJ-AUTO", "Bajaj Auto", "Automobile"),
)


def labels() -> list[str]:
    """Return symbol labels for UI controls.

    Returns:
        List of stock labels.
    """

    return [item.label for item in NSE_SYMBOLS]


def symbol_from_label(label: str) -> str:
    """Return symbol from a UI label.

    Args:
        label: Selected label.

    Returns:
        NSE symbol.
    """

    return label.split(" - ", maxsplit=1)[0]


def all_nse_labels(project_root: Path) -> list[str]:
    """Build searchable equity labels from local reference and universe files.

    Args:
        project_root: Absolute NATIP project directory.

    Returns:
        Deduplicated stock labels, falling back to the curated catalogue.
    """
    names = {item.symbol: item.name for item in NSE_SYMBOLS}
    sources = (
        project_root / "config/reference/nse_securities.csv",
        project_root / "stocks_universe_2026_08.csv",
        project_root / "stocks_universe_v5_smallcap250_2026_08_11.csv",
        project_root / "data/nifty_smallcap250_constituents.csv",
    )
    for path in sources:
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if row.get("active", "true").strip().lower() in {"false", "0"}:
                        continue
                    symbol = str(
                        row.get("symbol") or row.get("Ticker") or row.get("Symbol") or ""
                    ).strip().upper().removesuffix(".NS")
                    if not symbol:
                        continue
                    name = str(
                        row.get("Company") or row.get("Company Name") or ""
                    ).strip()
                    names[symbol] = name or names.get(symbol, "")
        except (OSError, csv.Error):
            continue
    return [f"{symbol} - {name}" if name else symbol for symbol, name in sorted(names.items())]


def sector_for_symbol(symbol: str) -> str:
    """Return known sector for a symbol.

    Args:
        symbol: NSE symbol.

    Returns:
        Sector name or unknown.
    """

    for item in NSE_SYMBOLS:
        if item.symbol == symbol:
            return item.sector
    return "Unknown"
