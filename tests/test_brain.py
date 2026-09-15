"""Sanity checks for prompt building — no API key or network needed.

Run with: python -m pytest tests/  (or just: python tests/test_brain.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_calling_agent.brain import build_system_prompt  # noqa: E402
from ai_calling_agent.config import load_agent_config, load_call_scripts  # noqa: E402


def test_build_system_prompt_includes_persona_and_objections():
    cfg = load_agent_config()
    scripts = load_call_scripts()
    prompt = build_system_prompt(cfg, scripts)

    assert cfg.name in prompt
    assert cfg.company in prompt
    assert "I'm not interested" in prompt
    assert "short sentences" in prompt.lower()


def test_all_scripts_have_required_fields():
    scripts = load_call_scripts()
    required = {"goal", "opening_line", "qualifying_questions", "closing_line"}
    for key, script in scripts.items():
        if key in ("objection_handling", "global_guidelines"):
            continue
        missing = required - script.keys()
        assert not missing, f"script '{key}' missing fields: {missing}"


def test_unknown_active_script_raises_clear_error():
    cfg = load_agent_config()
    cfg.active_script = "does_not_exist"
    scripts = load_call_scripts()
    try:
        build_system_prompt(cfg, scripts)
        raise AssertionError("expected KeyError for unknown script")
    except KeyError as e:
        assert "does_not_exist" in str(e)


if __name__ == "__main__":
    test_build_system_prompt_includes_persona_and_objections()
    test_all_scripts_have_required_fields()
    test_unknown_active_script_raises_clear_error()
    print("All tests passed.")
