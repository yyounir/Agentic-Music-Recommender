"""Launch the Streamlit UI.

Equivalent to `streamlit run src/main.py`, but it checks the environment first and
explains anything that would otherwise show up as a confusing error in the browser.

    python run_app.py
    python run_app.py --port 8502
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def preflight() -> bool:
    """Check dependencies and data files. Returns False if the app cannot start."""
    ok = True

    missing = []
    for module, package in (
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("pydantic", "pydantic"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"[x] Missing required packages: {', '.join(missing)}")
        print("    Fix with: pip install -r requirements.txt")
        ok = False
    else:
        print("[ok] Core dependencies installed")

    for relative in ("data/songs.csv", "data/editorial_notes.json"):
        if not (PROJECT_ROOT / relative).exists():
            print(f"[x] Missing data file: {relative}")
            ok = False
    if ok:
        print("[ok] Catalog and editorial notes found")

    # Gemini is optional: the app runs fully on the deterministic engine.
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.llm_client import DEFAULT_GEMINI_MODEL, load_dotenv

        loaded = load_dotenv()
        has_key = bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if has_key:
            source = ".env" if "GEMINI_API_KEY" in loaded else "environment"
            print(f"[ok] GEMINI_API_KEY found ({source})")
            print(f"     Default model: {DEFAULT_GEMINI_MODEL}")
            print("     Pick the engine and model in the app's sidebar.")
            print("     Free-tier quotas are per-model per-day; if one is")
            print("     exhausted the app falls back and tells you which model")
            print("     answered.")
        else:
            print("[--] No GEMINI_API_KEY found - the app will run on the")
            print("     deterministic offline engine (instant, no network).")
            print("     To enable Gemini, create a .env file containing:")
            print("         GEMINI_API_KEY=your-key-here")
    except ImportError:
        pass  # dependency problem already reported above

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Streamlit UI.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print(" Agentic Music Recommender - starting Streamlit UI")
    print("=" * 62)
    if not preflight():
        print("\nStartup aborted - fix the items marked [x] above.")
        return 1

    print("-" * 62)
    print(f" Opening http://localhost:{args.port}")
    print(" Press Ctrl+C to stop.")
    print("-" * 62)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "src" / "main.py"),
        "--server.port",
        str(args.port),
    ]
    if args.no_browser:
        command += ["--server.headless", "true"]

    try:
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
