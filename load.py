"""Load MALT runs into fxtr: `uv run python load.py [--tasks a,b] [--name NAME]`.

Reads the `default` config of `metr-evals/malt-public` (parquet files downloaded to
risk_behavior_motive/data/raw/malt/data), converts each run of the chosen tasks to a behaviors
`AgentRun`, and stores them in the project array `malt_<name>` keyed by [task, run]. It also stores
each task's statement (the first user message, as a `TextEntity`) in `malt_<name>_tasks` keyed by
[task], and runs `malt.index_runs` on a snapshot of the runs so they can be browsed in the viewer.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import anyio
import duckdb

from behaviors.types import AgentRun, TextEntity
from fxtr.core.entities import BoundID
from fxtr.entity_defns.array import Array
from fxtr.project.client import JobNotFinishedError
from fxtr.project.running import open_local_client, run_job
from malt_fxtr.convert import convert, task_statement
from malt_fxtr.runs import index_runs

DATA = Path.home() / "risk_behavior_motive/data/raw/malt/data"

# The pilot for ignores_task_instructions: every task with a manually reviewed run carrying that label.
PILOT_TASKS = [
    "ai_rd_restricted_mlm/main",
    "ai_rd_nanogpt_chat_rl/main",
    "debug_small_libs/orm_allbugs",
    "debug_small_libs/orm_somebugs",
    "machine_learning_local/cifar10",
]


def read_rows(tasks: list[str]) -> list[tuple[dict, list]]:
    con = duckdb.connect()
    return con.execute(
        f"select metadata, samples from read_parquet('{DATA}/*.parquet') "
        "where metadata.task_id in (select unnest(?)) order by metadata.task_id, metadata.run_id",
        [tasks],
    ).fetchall()


async def main(args: argparse.Namespace) -> None:
    tasks = args.tasks.split(",") if args.tasks else PILOT_TASKS
    rows = read_rows(tasks)
    missing = set(tasks) - {meta["task_id"] for meta, _ in rows}
    if missing:
        raise SystemExit(f"no runs for tasks: {sorted(missing)}")

    runs: list[tuple[str, AgentRun]] = []
    statements: dict[str, Counter] = {}
    for meta, samples in rows:
        runs.append((meta["task_id"], convert(meta, samples)))
        if (text := task_statement(samples)) is not None:
            statements.setdefault(meta["task_id"], Counter())[text] += 1
    print(f"converted {len(runs)} runs from {len(tasks)} tasks")
    for task, counts in statements.items():
        if len(counts) > 1:
            print(f"  {task}: {len(counts)} distinct task statements; using the most common ({counts.most_common(1)[0][1]} runs)")

    array = f"malt_{args.name}"
    async with open_local_client(__file__, max_concurrency=16) as client:
        refs = [((task, run.name or ""), await client.store(run)) for task, run in runs]
        items = Array.from_items(refs, ("[task: str, run: str]", BoundID[AgentRun]))
        task_refs = [((task,), await client.store(TextEntity(text=counts.most_common(1)[0][0]))) for task, counts in statements.items()]
        task_items = Array.from_items(task_refs, ("[task: str]", BoundID[TextEntity]))
        existing = {info.name for info in await client.list_arrays()}
        for name, value, schema in (
            (array, items, ("[task: str, run: str]", BoundID[AgentRun])),
            (f"{array}_tasks", task_items, ("[task: str]", BoundID[TextEntity])),
        ):
            if name not in existing:
                await client.create_array(name, schema)
            await client.replace_array(name, value)
            print(f"project array {name}: {len(value)} rows")
        snapshot = await client.snapshot_array(array)
        await run_job(client, index_runs, {"runs": snapshot.array}, root=f"index-{args.name}-v1")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="", help="comma-separated MALT task IDs (default: the pilot tasks)")
    p.add_argument("--name", default="pilot_iti")
    try:
        anyio.run(main, p.parse_args())
    except JobNotFinishedError:
        sys.exit(1)
