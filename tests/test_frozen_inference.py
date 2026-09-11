from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import app.probability.frozen_inference as inference
from run_final_signal_threshold_research import (
    _exclude_vedl_demerger_contamination,
    _prepare_dataset,
)


def test_frozen_feature_order_matches_config() -> None:
    config = json.loads(Path("reports/probability/frozen_model_config.json").read_text())

    assert inference.load_frozen_objects()[0]["feature_list"] == config["feature_list"]


def test_frozen_hashes_match_expected_identifiers() -> None:
    config = json.loads(Path("reports/probability/frozen_model_config.json").read_text())
    rules = json.loads(Path("reports/probability/frozen_signal_rules.json").read_text())

    assert config["config_hash"] == inference.EXPECTED_CONFIG_HASH
    assert rules["rules_hash"] == inference.EXPECTED_RULES_HASH


def test_percentiles_use_same_date_reference_universe() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "symbol": ["A.NS", "B.NS", "A.NS"],
            "p_outperform": [0.2, 0.8, 0.1],
            "buy_sigmoid_probability": [0.7, 0.6, 0.9],
            "ranker_score": [1.0, 2.0, 0.0],
        }
    )

    frame["p_outperform_percentile"] = inference._same_date_percentile(frame, "p_outperform")
    frame["buy_probability_percentile"] = inference._same_date_percentile(
        frame, "buy_sigmoid_probability"
    )

    jan1 = frame[frame["Date"].eq(pd.Timestamp("2026-01-01"))].set_index("symbol")
    assert jan1.loc["B.NS", "p_outperform_percentile"] == 1.0
    assert jan1.loc["A.NS", "buy_probability_percentile"] == 1.0


def test_frozen_rule_application_matches_json() -> None:
    rules = json.loads(Path("reports/probability/frozen_signal_rules.json").read_text())

    buy_row = pd.Series({"p_outperform_percentile": 0.99, "buy_probability_percentile": 0.99})
    accumulate_row = pd.Series(
        {"p_outperform_percentile": 0.80, "buy_probability_percentile": 0.50}
    )
    none_row = pd.Series({"p_outperform_percentile": 0.79, "buy_probability_percentile": 0.99})

    assert rules["buy"]["rule_id"] == "FROZEN_BUY_PRUNED_TOP_1PCT_AND_BINARY_SIGMOID_TOP_1PCT"
    assert inference.apply_frozen_signal_rules(buy_row) == "BUY"
    assert inference.apply_frozen_signal_rules(accumulate_row) == "ACCUMULATE"
    assert inference.apply_frozen_signal_rules(none_row) == "ACCUMULATE"


def test_tuned_only_can_never_generate_buy() -> None:
    rules = {
        "buy": {
            "primary_classifier_percentile_min": 0.99,
            "current_binary77_percentile_min": 0.995,
        },
        "tuned_binary77_confidence": {"confirmation_percentile_min": 0.995},
    }
    row = pd.Series(
        {
            "p_outperform_percentile": 0.999,
            "buy_probability_percentile": 0.20,
            "tuned_binary77_percentile": 1.0,
        }
    )

    result = inference.apply_tuned_binary77_confidence_layer(
        row, base_recommendation="ACCUMULATE", confidence_rules=rules
    )

    assert result["recommendation"] == "ACCUMULATE"
    assert result["tuned_status"] == "RESEARCH_ONLY_TUNED_SIGNAL"
    assert result["tuned_binary77_confirmation_pass"] is True


def test_current_buy_with_tuned_confirm_is_high_confidence() -> None:
    rules = {
        "buy": {
            "primary_classifier_percentile_min": 0.99,
            "current_binary77_percentile_min": 0.995,
        },
        "tuned_binary77_confidence": {"confirmation_percentile_min": 0.995},
    }
    row = pd.Series(
        {
            "p_outperform_percentile": 0.999,
            "buy_probability_percentile": 0.996,
            "tuned_binary77_percentile": 0.996,
        }
    )

    result = inference.apply_tuned_binary77_confidence_layer(
        row, base_recommendation="BUY", confidence_rules=rules
    )

    assert result["recommendation"] == "BUY"
    assert result["confidence"] == "HIGH"
    assert result["current_binary77_pass"] is True


def test_current_buy_without_tuned_confirm_is_medium_confidence() -> None:
    rules = {
        "buy": {
            "primary_classifier_percentile_min": 0.99,
            "current_binary77_percentile_min": 0.995,
        },
        "tuned_binary77_confidence": {"confirmation_percentile_min": 0.995},
    }
    row = pd.Series(
        {
            "p_outperform_percentile": 0.999,
            "buy_probability_percentile": 0.996,
            "tuned_binary77_percentile": 0.50,
        }
    )

    result = inference.apply_tuned_binary77_confidence_layer(
        row, base_recommendation="BUY", confidence_rules=rules
    )

    assert result["recommendation"] == "BUY"
    assert result["confidence"] == "MEDIUM"
    assert result["tuned_binary77_confirmation_pass"] is False


def test_tuned_confirmation_does_not_change_existing_buy_logic() -> None:
    rules = {
        "buy": {
            "primary_classifier_percentile_min": 0.99,
            "current_binary77_percentile_min": 0.995,
        },
        "tuned_binary77_confidence": {"confirmation_percentile_min": 0.995},
    }
    tuned_false = pd.Series(
        {
            "p_outperform_percentile": 0.999,
            "buy_probability_percentile": 0.996,
            "tuned_binary77_percentile": 0.0,
        }
    )
    tuned_true = tuned_false.copy()
    tuned_true["tuned_binary77_percentile"] = 1.0

    assert inference.current_buy_rule_pass(tuned_false, confidence_rules=rules)
    assert inference.current_buy_rule_pass(tuned_true, confidence_rules=rules)
    assert inference.apply_frozen_signal_rules(tuned_false, confidence_rules=rules) == "BUY"
    assert inference.apply_frozen_signal_rules(tuned_true, confidence_rules=rules) == "BUY"


def test_prediction_path_does_not_import_training_fitters() -> None:
    source = Path("predict_stock.py").read_text()

    assert "_fit_classifier" not in source
    assert "train" not in source.lower()


def test_invalid_ticker_rejected() -> None:
    with pytest.raises(ValueError):
        inference.validate_ticker("reliance")


def test_public_prediction_dict_contains_streamlit_keys() -> None:
    prediction = inference.FrozenPrediction(
        ticker="RELIANCE.NS",
        signal="ACCUMULATE",
        final_recommendation="ACCUMULATE",
        signal_date="2026-08-07",
        latest_adjusted_price=1334.8,
        p_outperform=0.21,
        p_neutral=0.62,
        p_underperform=0.17,
        p_outperform_percentile=0.11,
        buy_raw_probability=0.22,
        buy_sigmoid_probability=0.23,
        buy_probability_percentile=0.20,
        primary_xgb_pass=False,
        current_binary77_pass=False,
        tuned_binary77_confirmation_pass=False,
        current_binary77_percentile=0.20,
        tuned_binary77_raw_probability=0.24,
        tuned_binary77_sigmoid_probability=0.25,
        tuned_binary77_percentile=0.21,
        tuned_status="NO_TUNED_CONFIRMATION",
        p_sell=0.19,
        sell_percentile=0.28,
        ranker_percentile=0.22,
        model_agreement="0/2 ACCUMULATE-positive components",
        market_regime="Sideways",
        sector="Energy",
        horizon_trading_days=20,
        confidence_evidence_level="MEDIUM",
        config_hash=inference.EXPECTED_CONFIG_HASH,
        rules_hash=inference.EXPECTED_RULES_HASH,
        confidence_config_hash="candidate-config-hash",
        confidence_rules_hash="candidate-rules-hash",
        in_training_universe=True,
        warning=None,
        missing_imputed_feature_warnings=["1 frozen feature was median-imputed"],
        top_supporting_feature_explanations=[],
    )

    payload = inference.prediction_to_dict(prediction)

    assert payload["recommendation"] == "ACCUMULATE"
    assert payload["latest_price"] == 1334.8
    assert payload["Pruned P_Outperform"] == 0.21
    assert payload["Binary BUY sigmoid probability"] == 0.23
    assert payload["Current Binary77 pass"] is False
    assert payload["Tuned Binary77 confirmation pass"] is False
    assert payload["Tuned Binary percentile"] == 0.21
    assert payload["XGBRanker percentile diagnostic"] == 0.22
    assert payload["confidence"] == "MEDIUM"
    assert payload["warnings"] == ["1 frozen feature was median-imputed"]
    assert inference.PREDICTION_HISTORY.is_absolute()


def test_final_recommendation_enum_has_only_three_states() -> None:
    row = pd.Series({"p_outperform_percentile": 0.1, "buy_probability_percentile": 0.2})

    assert inference.apply_frozen_signal_rules(row) in {"BUY", "ACCUMULATE", "SELL"}


def test_vedl_structural_break_cleanup_is_consistent() -> None:
    clean = _prepare_dataset(
        pd.DataFrame(
            {
                "Date": pd.date_range("2026-03-01", periods=90, freq="B"),
                "symbol": ["VEDL.NS"] * 90,
                "Close": range(90, 180),
                "excess_return": [0.0] * 90,
                "future_stock_return": [0.0] * 90,
                "future_nifty_return": [0.0] * 90,
                "label": [0] * 90,
            }
        )
    )

    filtered, cleanup = _exclude_vedl_demerger_contamination(clean)

    assert not cleanup.empty
    assert cleanup["EventType"].eq("CORPORATE_ACTION_STRUCTURAL_BREAK").all()
    assert cleanup["Action"].eq("EXCLUDE").all()
    assert len(filtered) + len(cleanup) == len(clean)
