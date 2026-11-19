#!/usr/bin/env python3
"""Run the Retrieval Evals Scorecard and write the HTML report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_scorecard.runner import Settings, load_dotenv, run_scorecard, write_result  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Compare retrieval configs and write an HTML scorecard.")
    parser.add_argument("--mode", choices=("local", "live", "auto"), default=None, help="Default: local, or SCORECARD_MODE.")
    parser.add_argument("--provider", choices=("mock", "nvidia", "auto"), default=None, help="Default: LLM_PROVIDER or auto.")
    parser.add_argument("--out", default=str(ROOT / "out" / "scorecard.html"), help="HTML report path. JSON is written beside it.")
    args = parser.parse_args(argv)
    settings = Settings.from_env(mode=args.mode, provider=args.provider)
    result = run_scorecard(settings)
    out = Path(args.out)
    write_result(result, out)
    print(f"mode={result.mode} embeddings={result.embedding_backend}")
    for item in result.config_scores:
        print(
            f"{item.config.name}: recall@k={item.recall_at_k:.3f} "
            f"mrr={item.mrr:.3f} faithfulness={item.faithfulness:.3f}"
        )
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
