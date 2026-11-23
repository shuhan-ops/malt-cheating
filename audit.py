"""Launch the stated-rules audit over stored MALT runs.

    uv run python audit.py --dry-run          # show what would run, call no model
    uv run python audit.py                    # the manually reviewed runs of the pilot tasks
    uv run python audit.py --all              # every run in the array, reviewed or not

Reads the rubrics from rubrics/. Models are named `anthropic/...`, `openai/...` or
`openrouter/...`; their API keys must be in .env.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import anyio
import duckdb

from behaviors.judging.defaults import DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY
from behaviors.types import AgentRun, ReasoningParams, SamplingParams, TextEntity
from fxtr.core.entities import BoundID
from fxtr.entity_defns.array import Array
from fxtr.project.client import JobNotFinishedError
from fxtr.project.running import open_local_client, run_job
from malt_fxtr.audit import SamplingParamsEntity, audit_rules

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "risk_behavior_motive/data/raw/malt/data"


def reviewed_run_ids() -> set[str]:
    """Run IDs MALT reviewed by hand, read from the parquet metadata (faster than loading runs)."""
    rows = duckdb.connect().execute(
        f"select metadata.run_id from read_parquet('{DATA}/*.parquet') where metadata.manually_reviewed"
    ).fetchall()
    return {str(r[0]) for r in rows}


async def main(args: argparse.Namespace) -> None:
    async with open_local_client(__file__, max_concurrency=args.concurrency) as client:
        snapshot = await client.snapshot_array(args.array)
        full = await client.load(snapshot.array)
        reviewed = reviewed_run_ids()
        items = [(key, ref) for key, ref in full.items() if args.all or key[1] in reviewed]
        print(f"{len(items)} runs: " + ", ".join(f"{t} {n}" for t, n in sorted(Counter(k[0] for k, _ in items).items())))
        if args.dry_run:
            print(f"extract with {args.extract_model}, judge with {args.judge_model}; dry run, nothing launched")
            return
        runs = Array.from_items(items, ("[task: str, run: str]", BoundID[AgentRun]))

        async def text(t: str) -> Array:
            return Array.scalar(await client.store(TextEntity(text=t)), BoundID[TextEntity])

        async def sampling(effort: str) -> Array:
            ref = await client.store(SamplingParamsEntity(params=SamplingParams(max_tokens=16000, reasoning=ReasoningParams(effort=effort))))
            return Array.scalar(ref, BoundID[SamplingParamsEntity])

        inputs = {
            "runs": runs,
            "max_opening_chars": Array.scalar(30000, int),
            "max_tool_chars": Array.scalar(args.max_tool_chars, int),
            "extract_model": Array.scalar(args.extract_model, str),
            "extract_sampling": await sampling("medium"),
            "extract_rubric": await text((HERE / "rubrics/extract_rules.md").read_text()),
            "judge_model": Array.scalar(args.judge_model, str),
            "judge_sampling": await sampling("high"),
            "judge_rubric": await text((HERE / "rubrics/judge_ignores_instructions.md").read_text()),
            "instructions": Array.scalar(await client.store(DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY), BoundID[TextEntity]),
        }
        await run_job(client, audit_rules, inputs, root=f"audit-rules-{'all' if args.all else 'reviewed'}-v1")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--array", default="malt_pilot_iti")
    p.add_argument("--all", action="store_true", help="include runs MALT did not review by hand")
    p.add_argument("--extract-model", default="anthropic/claude-sonnet-5-5")
    p.add_argument("--judge-model", default="anthropic/claude-opus-5-5")
    p.add_argument("--max-tool-chars", type=int, default=3000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    try:
        anyio.run(main, p.parse_args())
    except JobNotFinishedError:
        sys.exit(1)
