"""MALT runs in fxtr: a workflow that indexes a set of stored agent runs.

The launcher (`load.py`) converts each MALT row into a behaviors `AgentRun`, stores it, and passes
the references in as `runs`, keyed by [task, run] (sparse: each task has its own runs). This
workflow flattens each run's MALT metadata and counts into a row, and returns a `RunSet` for the
viewer. Later experiments (the audit) take the same `runs` array.
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

from behaviors.types import AgentRun, AssistantMessage, ContentReasoning, ContentToolUse
from fxtr.core.entities import BoundID
from fxtr.entity_defns.array import Array
from fxtr.entity_defns.dataclass_entity import DataclassEntity
from fxtr.experiment.handles import ArrayHandle
from fxtr.experiment.steps import StepContext, step
from fxtr.experiment.workflows import WorkflowContext, workflow


class RunStats(TypedDict):
    """One MALT run's metadata and counts, stored inline as a value in an array."""

    model: str | None
    labels: list[str]  # MALT's behavior labels for the run
    manually_reviewed: bool
    run_source: str | None  # "unprompted" (natural) or "prompted" (told to misbehave)
    has_chain_of_thought: bool
    samples: int  # model calls MALT recorded
    side_calls: int  # calls outside the main conversation (e.g. a rater), left out of the transcript
    extra_candidates: int  # sampled completions beyond the first, across all calls
    messages: int
    tool_calls: int
    reasoning_blocks: int


@dataclass(frozen=True)
class RunSet(DataclassEntity, fxtr_type="local.malt_fxtr.RunSet.v1"):
    """A set of MALT agent runs and their metadata, both arrays keyed by [task, run]."""

    runs: BoundID[Array[BoundID[AgentRun]]]
    stats: BoundID[Array[RunStats]]


@step(name="malt.run_stats")
async def run_stats(context: StepContext, run: AgentRun) -> RunStats:
    """Flatten a run's MALT metadata and count its messages, tool calls, and reasoning."""
    meta = run.metadata
    messages = [m for t in run.transcripts for m in t.messages]
    contents = [c for m in messages if isinstance(m, AssistantMessage) for c in m.content]

    def as_int(v: object) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) else 0

    labels = meta.get("labels")
    return RunStats(
        model=meta.get("model") if isinstance(meta.get("model"), str) else None,
        labels=[str(x) for x in labels] if isinstance(labels, list) else [],
        manually_reviewed=bool(meta.get("manually_reviewed")),
        run_source=meta.get("run_source") if isinstance(meta.get("run_source"), str) else None,
        has_chain_of_thought=bool(meta.get("has_chain_of_thought")),
        samples=as_int(meta.get("samples")),
        side_calls=as_int(meta.get("side_calls")),
        extra_candidates=as_int(meta.get("extra_candidates")),
        messages=len(messages),
        tool_calls=sum(isinstance(c, ContentToolUse) for c in contents),
        reasoning_blocks=sum(isinstance(c, ContentReasoning) for c in contents),
    )


@workflow(name="malt.index_runs")
async def index_runs(
    context: WorkflowContext,
    runs: Annotated[ArrayHandle[BoundID[AgentRun]], "[task: str, run: str]"],
) -> RunSet:
    """List each MALT run with its model, labels, and message counts."""
    stats = context.run_step("run_stats", run_stats, {"run": runs}, map_over=["task", "run"])
    return RunSet(
        runs=await context.store(await runs.observe()),
        stats=await context.store(await stats.observe()),
    )
