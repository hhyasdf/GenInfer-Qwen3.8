"""Phase gates for a generated engine. Stdlib only.

    python -m minisgl.gates --phase B4

Run from the engine directory. A failing check exits 1 and prints one line.
``--phase`` does not decide whether the engine may bind the agent port; that
stays in the supervisor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path.cwd()


def _section(text: str, name: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    seen = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if seen:
                break
            seen = stripped == f"[{name}]"
            continue
        if seen:
            out.append(line)
    return "\n".join(out)


def _key(text: str, section: str, key: str) -> str:
    for line in _section(text, section).splitlines():
        body = line.split("#", 1)[0].strip()
        if body.startswith(f"{key} ") or body.startswith(f"{key}="):
            return body.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def config_text() -> str:
    path = ROOT / "config.toml"
    if not path.is_file():
        return ""
    return path.read_text()


def check_not_toy() -> str | None:
    model = _key(config_text(), "engine", "model")
    if model in ("", "models/tiny-llama"):
        return "config.toml engine.model is still models/tiny-llama"
    return None


def _resolved_model(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    return model


def check_golden_model() -> str | None:
    golden = ROOT / "golden" / "logits.json"
    if not golden.is_file():
        return "golden/logits.json is missing"
    try:
        data = json.loads(golden.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"golden/logits.json is unreadable: {exc}"
    recorded = str(data.get("model", ""))
    model = _key(config_text(), "engine", "model")
    resolved = _resolved_model(model)
    if recorded not in (model, resolved):
        return f"golden model id {recorded!r} does not match config model {resolved!r}"
    return None


def check_origin_if_consumers() -> str | None:
    agents = ROOT / "AGENTS.md"
    if not agents.is_file():
        return "AGENTS.md is missing"
    dependent = False
    for line in agents.read_text().splitlines():
        if "Dependent consumers" not in line and "dependent consumers" not in line:
            continue
        low = line.lower()
        if "<yes/no" in low:
            continue
        if "yes" in low.split(":", 1)[-1]:
            dependent = True
    scenario = ROOT / "SCENARIO.md"
    if scenario.is_file():
        for line in scenario.read_text().splitlines():
            if line.lower().strip().startswith("- consumer:"):
                if "agent" in line.lower():
                    dependent = True
    if not dependent:
        return None
    command = _key(config_text(), "origin", "command")
    if not command.strip():
        return "dependent consumers are set but [origin].command is empty"
    return None


PHASES = {
    "B3": (check_not_toy,),
    "B4": (check_not_toy, check_golden_model),
    "B5": (check_not_toy, check_golden_model, check_origin_if_consumers),
    "T0": (check_not_toy,),
    "T1": (check_not_toy,),
    "T2": (check_not_toy, check_golden_model),
    "T3": (check_not_toy, check_golden_model),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minisgl.gates")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    args = parser.parse_args(argv)
    for check in PHASES[args.phase]:
        err = check()
        if err:
            print(f"{args.phase}: {err}", file=sys.stderr)
            return 1
    print(f"{args.phase}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
