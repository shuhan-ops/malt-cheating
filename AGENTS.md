# Agent notes for malt-fxtr

## Naming stages

Every step is named `<scope>_<verb>_<object>`:

- **scope** says what one call of the step sees:
  - `row`: one item (a run, a task), so the step is mapped over the array's dimensions.
  - `frame`: a whole array at once, so the step is not mapped (or is mapped over fewer dimensions
    than it reads).
- **verb** says what the step does, from the list below.
- **object** says what it acts on or produces.

Use the same name in both places a step is named: its registered name,
`@step(name="malt.<scope>_<verb>_<object>")`, and its local name in the workflow,
`context.run_step("<scope>_<verb>_<object>", ...)`. When one step runs twice in a workflow, add
what tells the runs apart: `row_judge_rules_full`, `row_judge_rules_actions_only`.

| Verb | Model call? | What the step does | Examples |
|---|---|---|---|
| `load` | no | Turn outside data (files, datasets, APIs) into stored entities. Usually in a launcher, since steps never read files. | `load.py` |
| `transform` | no | Reshape: select, cap, strip, flatten, format. | `row_transform_compact_run`, `row_transform_actions_only` |
| `scan` | no | Find matches with fixed rules (regexes, counts, thresholds) and report hits. | `row_scan_keywords`, `row_scan_tool_failures` |
| `aggregate` | no | Combine many rows into one: counts, rates, agreement tables. | `frame_aggregate_agreement` |
| `extract` | yes | Pull structured facts out of text, with no verdict on them. | `row_extract_rules` |
| `label` | yes | Put each item (a turn, a message) in one category of a fixed set. | `row_label_turns` |
| `judge` | yes | Give verdicts against a rubric, with cited evidence. | `row_judge_rules_full` |
| `summarize` | yes | Condense text for a human reader. | `row_summarize_turns` |

A step that fits none of these verbs is named `<scope>_custom_<verb>_<object>` (e.g.
`frame_custom_explore_outliers` for an agent that investigates a set of runs), and its docstring
says what the verb means.

**Workflows are named for what they produce, as a noun**: `malt.run_index`, `malt.rules_audit`.
Steps start with a scope and a verb and workflows are nouns, so a name tells you which it is.

**Renaming a step changes its cache address**, so the next launch recomputes it and everything
after it. Rename before a rerun you need anyway, or adopt the old job's cache afterwards
(`fxtr cache adopt JOB`) and say which results it covers.
