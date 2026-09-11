"""Curated NSE symbol catalog for the dashboard."""

from dataclasses import dataclass


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
