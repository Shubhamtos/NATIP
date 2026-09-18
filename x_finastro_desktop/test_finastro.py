from finastro import classify


def test_financial_astrology_market_call():
    result = classify("Saturn transit may keep Nifty bearish this week. #finastro")
    assert result["is_finastro"] is True
    assert "saturn" in result["astro_terms"]
    assert "nifty" in result["market_terms"]
    assert result["direction"] == "Bearish"


def test_non_astro_post_is_rejected():
    result = classify("Nifty rallied strongly after earnings.")
    assert result["is_finastro"] is False


def test_bullish_astro_post():
    result = classify("Jupiter transit suggests a bullish phase for gold.")
    assert result["is_finastro"] is True
    assert "jupiter" in result["astro_terms"]
    assert "gold" in result["market_terms"]
    assert result["direction"] == "Bullish"
