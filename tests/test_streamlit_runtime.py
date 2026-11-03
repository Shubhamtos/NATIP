"""Regression checks for bundled-runtime reload corruption in Streamlit."""

import enum
import tomllib
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError
from streamlit.folder_black_list import FolderBlackList

from app.decision.rule_engine.dual_listed_darvas import DarvasState, DualListedDarvasResult

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_standard_library_is_excluded_from_app_reload():
    settings = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["server"]
    assert settings["fileWatcherType"] == "none"
    assert settings["runOnSave"] is False
    blacklist = FolderBlackList(settings["folderWatchBlacklist"])
    # Use explicit project paths as well as the executing interpreter's runtime.
    assert blacklist.is_blacklisted(str(ROOT / "python-standalone/python/lib/python3.12/enum.py"))
    assert blacklist.is_blacklisted(str(ROOT / "python-standalone/python/lib/python3.12/typing.py"))
    assert not blacklist.is_blacklisted(str(ROOT / "app/dashboard/streamlit_app.py"))
    for module in (enum, typing):
        path = Path(module.__file__).resolve()
        if path.is_relative_to(ROOT):
            assert blacklist.is_blacklisted(str(path))


def test_darvas_enum_keeps_strict_validation_and_json_schema():
    schema = DualListedDarvasResult.model_json_schema()
    assert schema["$defs"]["DarvasState"]["enum"] == [state.value for state in DarvasState]
    result = DualListedDarvasResult.model_validate(
        {
            "symbol": "RELIANCE",
            "darvas_state": "BREAKOUT_CONFIRMED",
        }
    )
    assert result.darvas_state is DarvasState.BREAKOUT_CONFIRMED
    assert DualListedDarvasResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        DualListedDarvasResult.model_validate({"symbol": "RELIANCE", "darvas_state": "MADE_UP"})
