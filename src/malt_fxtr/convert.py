"""Convert one MALT run (a row of `metr-evals/malt-public`, the list format) into a behaviors `AgentRun`.

A MALT row is one run: `metadata` (model, task_id, run_id, labels, ...) and `samples`, one per model
call. Each sample has the `input` messages sent to the API and the `output` completions it returned
(a list of N candidates, each a list of messages: often a `reasoning_completion` then a
`completion`). Input messages carry `node_id` / `parent_node_id` in their metadata; outputs do not.

No single sample holds the whole run. The scaffolds trim the middle of long contexts, so later
samples drop earlier messages. We rebuild the transcript by walking the samples in order:

- each input message not seen before (by `node_id`, or by content when it has none) is appended;
- each sample's chosen output is appended after that sample's inputs. The chosen candidate is the
  one whose content reappears as an input of a later sample, else the first. A later input that
  repeats an output already appended is skipped.

Mapping to behaviors types:

- `system` -> SystemMessage (a later system message, from scaffolds that send a fresh prompt per
  call, becomes a UserMessage marked as such);
- `user` -> UserMessage;
- `assistant` with `completion_type == "reasoning_completion"` -> ContentReasoning, merged into the
  next assistant message; other assistant messages -> ContentText, plus a ContentToolUse for a
  `function_call` (arguments parsed as JSON);
- `function` -> ToolMessage answering the most recent unanswered call (a result with no call becomes
  a UserMessage marked as tool output; a call with no result gets a placeholder result).

Only the launcher calls this: steps never read files.
"""

import json
from typing import Any

from behaviors.types import (
    AgentRun,
    AssistantMessage,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    SystemMessage,
    ToolMessage,
    Transcript,
    UserMessage,
)


def _fingerprint(m: dict[str, Any]) -> tuple:
    fc = m.get("function_call") or {}
    return (m.get("role"), (m.get("content") or "").strip(), fc.get("name"), fc.get("arguments"))


def linearize(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The run's messages in order, deduplicated across samples, with counts of what was dropped."""
    out: list[dict[str, Any]] = []
    seen_nodes: set[int] = set()
    seen_prints: set[tuple] = set()
    pending: list[tuple[int, list[list[dict[str, Any]]]]] = []  # (insert position, candidates)
    stats = {"samples": len(samples), "extra_candidates": 0, "unmatched_outputs": 0}

    def resolve_pending(upcoming: list[dict[str, Any]]) -> None:
        # Pick, for each pending sample, the candidate that a later input repeats; else the first.
        upcoming_prints = {_fingerprint(m) for m in upcoming}
        offset = 0
        for pos, candidates in pending:
            chosen = next(
                (c for c in candidates if any(_fingerprint(m) in upcoming_prints for m in c if m.get("role") == "assistant")),
                None,
            )
            if chosen is None:
                chosen = candidates[0]
                stats["unmatched_outputs"] += 1
            new = [m for m in chosen if _fingerprint(m) not in seen_prints]
            for m in new:
                seen_prints.add(_fingerprint(m))
            out[pos + offset : pos + offset] = new
            offset += len(new)
        pending.clear()

    for sample in samples:
        inputs = list(sample.get("input") or [])
        resolve_pending(inputs)
        for m in inputs:
            node = (m.get("metadata") or {}).get("node_id")
            fp = _fingerprint(m)
            if node is not None:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                if fp in seen_prints and m.get("role") == "assistant":
                    continue  # an output we already appended
            elif fp in seen_prints:
                continue
            seen_prints.add(fp)
            out.append(m)
        candidates = [c if isinstance(c, list) else [c] for c in (sample.get("output") or [])]
        if candidates:
            stats["extra_candidates"] += len(candidates) - 1
            pending.append((len(out), candidates))
    resolve_pending([])
    return out, stats


def _text(text: str) -> tuple[ContentText]:
    return (ContentText(text=text),)


def to_messages(raw: list[dict[str, Any]]) -> list[Any]:
    """behaviors messages for MALT messages in transcript order."""
    messages: list[Any] = []
    reasoning: list[str] = []  # reasoning waiting for its completion
    open_calls: list[str] = []  # ids of calls with no result yet
    n_calls = 0

    def close_calls() -> None:
        while open_calls:
            messages.append(ToolMessage(content=_text("[no tool result recorded]"), tool_call_id=open_calls.pop(0), is_error=None))

    def flush_reasoning() -> None:
        if reasoning:
            messages.append(AssistantMessage(content=tuple(ContentReasoning(text=r, form="full") for r in reasoning)))
            reasoning.clear()

    for i, m in enumerate(raw):
        role = m.get("role")
        content = m.get("content") or ""
        meta = m.get("metadata") or {}
        if role == "assistant" and meta.get("completion_type") == "reasoning_completion":
            close_calls()
            if content.strip():
                reasoning.append(content.strip())
            continue
        if role == "function":
            if open_calls:
                messages.append(ToolMessage(content=_text(content or "[empty tool output]"), tool_call_id=open_calls.pop(0), is_error=None))
            else:
                flush_reasoning()
                messages.append(UserMessage(content=_text(f"[tool output: {m.get('name') or 'unknown'}]\n{content}")))
            continue
        close_calls()
        if role == "assistant":
            parts: list[Any] = [ContentReasoning(text=r, form="full") for r in reasoning]
            reasoning.clear()
            if content.strip():
                parts.append(ContentText(text=content))
            fc = m.get("function_call") or {}
            if fc.get("name"):
                n_calls += 1
                call_id = f"call_{n_calls}"
                raw_args = fc.get("arguments") or "{}"
                try:
                    args, error = json.loads(raw_args), None
                    if not isinstance(args, dict):
                        args = {"arguments": args}
                except json.JSONDecodeError as e:
                    args, error = {"arguments": raw_args}, str(e)
                parts.append(ContentToolUse(id=call_id, name=fc["name"], input=args, parse_error=error))
                open_calls.append(call_id)
            if not parts:
                parts.append(ContentText(text="[empty assistant message]"))
            messages.append(AssistantMessage(content=tuple(parts)))
        elif role == "system":
            flush_reasoning()
            if i == 0:
                messages.append(SystemMessage(content=_text(content)))
            else:
                messages.append(UserMessage(content=_text(f"[system prompt for a new model call]\n{content}")))
        else:
            flush_reasoning()
            messages.append(UserMessage(content=_text(content)))
    flush_reasoning()
    close_calls()
    return messages


def main_thread(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The samples of the agent's main conversation, and how many side calls were left out.

    Some scaffolds (triframe) interleave calls with other prompts, such as a rater that gets the
    whole transcript pasted into one system message. The main thread is the samples that share the
    most common first message; if no first message repeats (each call is a fresh prompt), all
    samples are kept.
    """
    from collections import Counter

    def first(s: dict[str, Any]) -> str:
        ins = s.get("input") or []
        return (ins[0].get("content") or "") if ins else ""

    counts = Counter(first(s) for s in samples)
    top, n = counts.most_common(1)[0] if counts else ("", 0)
    if n <= 1:
        return samples, 0
    kept = [s for s in samples if first(s) == top]
    return kept, len(samples) - len(kept)


def convert(metadata: dict[str, Any], samples: list[dict[str, Any]]) -> AgentRun:
    """One MALT row as an AgentRun named by its run ID, with MALT's metadata and conversion counts."""
    thread, side_calls = main_thread(samples)
    raw, stats = linearize(thread)
    stats = {**stats, "samples": len(samples), "side_calls": side_calls}
    messages = to_messages(raw)
    meta = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in metadata.items()}
    return AgentRun(
        name=str(metadata["run_id"]),
        description=None,
        transcripts=(Transcript(name=None, description=None, messages=tuple(messages), metadata={}),),
        groups=(),
        metadata={**meta, **stats, "messages": len(messages)},
    )


def task_statement(samples: list[dict[str, Any]]) -> str | None:
    """The task as given to the agent: the first user message of the first sample."""
    for m in (samples[0].get("input") or []) if samples else []:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m["content"]
    return None
