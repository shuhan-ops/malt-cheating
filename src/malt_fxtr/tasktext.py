"""Find the task text in a run's opening, and group runs whose task texts are the same task.

Plain functions, used by the audit's steps (`row_transform_task_text`,
`frame_transform_cluster_task_texts`) and testable without fxtr.

The task text is what the task-giver wrote, without the scaffold around it:

- a `<task>...</task>` block, if a message before the agent's first action has one (triframe);
- otherwise the longest user message before the first action, with a leading scaffold phrase
  ending in a colon ("You are assigned this task:", "Your current task is the following:") removed;
- otherwise the system prompt.

The system prompt is skipped when a user message holds the task, because prompted runs put their
instructions (to sabotage, to sandbag) there, and those are not the task's rules.

Two runs share a cluster when one's task text, with whitespace collapsed, contains the other's.
Each cluster is represented by its longest text, so the extraction sees the most context.
"""

import hashlib
import re
from dataclasses import dataclass

_TASK_TAG = re.compile(r"<task>\s*(.*?)\s*</task>", re.S)
_LEAD = re.compile(r"^\s*[^\n:]{0,60}\b(task|following)\s*:\s*", re.I)
_SPACE = re.compile(r"\s+")


def task_text(opening: list[tuple[str, str]]) -> str:
    """The task text, from (role, text) pairs of the messages before the agent's first action."""
    for _, text in opening:
        if m := _TASK_TAG.search(text):
            return m.group(1).strip()
    users = [text for role, text in opening if role == "user" and text.strip()]
    if users:
        longest = max(users, key=len)
        if len(longest) >= 100:
            return _LEAD.sub("", longest, count=1).strip()
    systems = [text for role, text in opening if role == "system"]
    return (max(systems, key=len) if systems else "").strip()


def squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def text_id(text: str) -> str:
    return hashlib.sha256(squash(text).encode()).hexdigest()[:10]


@dataclass
class Cluster:
    id: str  # a hash of `text`
    text: str  # the most common member text that wraps no other member's (ties: the shortest)
    members: list[tuple[str, str]]  # (task, run) keys


def cluster(texts: dict[tuple[str, str], str]) -> list[Cluster]:
    """Group runs by task text containment, sorted by size.

    Texts are taken longest first; each joins the first group whose longest text contains it or is
    contained by it. A group's text is its most common member text, which is usually the task
    without the extra scaffold wording some runs wrap around it.
    """
    groups: list[tuple[str, list[tuple[str, str]]]] = []  # (longest squashed text, members)
    for key, text in sorted(texts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        s = squash(text)
        for longest, members in groups:
            if s in longest or longest in s:
                members.append(key)
                break
        else:
            groups.append((s, [key]))
    clusters = []
    for _, members in groups:
        counts: dict[str, int] = {}
        for key in members:
            counts[texts[key]] = counts.get(texts[key], 0) + 1
        # Prefer texts that wrap no other member's text: the bare task, not a scaffold around it.
        bare = [t for t in counts if not any(o != t and squash(o) in squash(t) for o in counts)]
        text = min(bare or counts, key=lambda t: (-counts[t], len(t)))
        clusters.append(Cluster(id=text_id(text), text=text, members=members))
    return sorted(clusters, key=lambda c: (-len(c.members), c.id))
