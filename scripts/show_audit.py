"""Print a stated-rules audit: `uv run python scripts/show_audit.py JOB_ID [--rules] [--runs a/b,c/d] [--json OUT]`.

Prints the summary, then (with --rules) each distinct set of extracted rules, then the per-run
verdicts of both judges for the runs where they or MALT flag a break (or the runs named in --runs).
--json writes every run's rows and judgments to a file for further analysis.
"""

import argparse
import json
from collections import defaultdict

import anyio
from fxtr.project.running import open_local_client

from malt_fxtr.audit import Audit


async def main(args: argparse.Namespace) -> None:
    async with open_local_client(__file__) as c:
        job = await c.job(args.job)
        audit = await c.load((await job.result()).item(), as_type=Audit)
        print(json.dumps(audit.summary, indent=1))

        if args.rules:
            rules = await c.load(audit.rules)
            by_text: dict[str, list[str]] = defaultdict(list)
            for key, ref in rules.items():
                by_text[(await c.load(ref)).text].append(f"{key[0]}/{key[1]}")
            for text, runs in sorted(by_text.items(), key=lambda kv: kv[1][0]):
                print(f"\n===== rules used by {len(runs)} runs (e.g. {runs[0]})\n{text}")

        rows = await c.load(audit.rows)
        full = await c.load(audit.judgments)
        acts = await c.load(audit.action_judgments)
        wanted = set(args.runs.split(",")) if args.runs else None
        dump = []
        for key, row in rows.items():
            name = f"{key[0]}/{key[1]}"
            jf, ja = await c.load(full[key]), await c.load(acts[key])
            dump.append({"run": name, "row": row, "full": jf.output, "actions": ja.output})
            flagged = row["malt_ignores_instructions"] or row["full"]["broke_stated_rule"] or row["actions"]["broke_stated_rule"]
            if (wanted is not None and name not in wanted) or (wanted is None and not flagged):
                continue
            print(f"\n===== {name}  {row['model']}  MALT={row['malt_labels']}")
            for label, j in (("full", jf), ("actions", ja)):
                o = j.output
                if o is None:
                    print(f"  [{label}] NO ACCEPTED OUTPUT")
                    continue
                print(f"  [{label}] broke={o['broke_stated_rule']}")
                for v in o["verdicts"]:
                    if v["verdict"] in ("attempted", "violated"):
                        print(f"    {v['rule_id']} {v['verdict']} ({v['confidence']}, ambiguous={v['ambiguous']}, disclosure={v['disclosure']}): {v['evidence'][:500]}")
                if o["circumvention"]:
                    print(f"    circumvention: {o['circumvention'][:300]}")
                if o["rules_feedback"]:
                    print(f"    rules_feedback: {o['rules_feedback'][:300]}")
            print(f"  discussion: {(jf.output or {}).get('discussion', '')[:900]}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dump, f, indent=1, default=str)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("--rules", action="store_true")
    p.add_argument("--runs", default="")
    p.add_argument("--json", default="")
    anyio.run(main, p.parse_args())
