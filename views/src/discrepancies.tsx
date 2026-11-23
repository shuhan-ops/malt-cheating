// The runs where MALT's `ignores_task_instructions` label and our two judges do not all agree
// (`malt.row_transform_discrepancy`, filtered in `malt.rules_audit`), one card each, to read by
// hand. A card shows the three calls side by side, then for each judge the rules it found broken:
// the rule, the task's words, the verdict, and the judge's evidence with its citations as chips
// that open the run at the cited message. A judge that found nothing shows its discussion instead,
// so a MALT-only miss shows why the judge cleared the run.

import { useState, type ReactNode } from "react";
import {
  EntityLink, isEntityID, rowsOf, unwrap, useEntity, useEntityLinks,
  type ArrayData, type EntityID,
} from "fxtr-view";
import { judgmentOf, type CitationData, type JudgmentData } from "fxtr-view-behaviors";
import { layout } from "fxtr-view-kit";
import styles from "./discrepancies.module.css";

const KINDS = ["ours only", "MALT only", "judges split"] as const;
type Kind = (typeof KINDS)[number];

const KIND_NOTE: Record<Kind, string> = {
  "ours only": "both judges flag, MALT does not",
  "MALT only": "MALT flags, neither judge does",
  "judges split": "the full and actions-only judges differ",
};

interface BrokenRule {
  readonly index: number;
  readonly rule_id: string;
  readonly rule: string;
  readonly quote: string;
  readonly verdict: string;
  readonly ambiguous: boolean;
  readonly confidence: string;
  readonly disclosure: string | null;
}

interface JudgeSide {
  readonly judged: boolean;
  readonly broke: boolean;
  readonly broken: readonly BrokenRule[];
}

interface Discrepancy {
  readonly kind: Kind;
  readonly model: string | null;
  readonly malt_labels: readonly string[];
  readonly malt: boolean;
  readonly full: JudgeSide;
  readonly actions: JudgeSide;
}

interface Item {
  readonly name: string;
  readonly task: string;
  readonly d: Discrepancy;
  readonly run: EntityID | undefined;
  readonly full: EntityID | undefined;
  readonly actions: EntityID | undefined;
}

const keyOf = (key: readonly unknown[]) => `${String(key[0])}/${String(key[1])}`;
const shortTask = (t: string) => t.replace(/^(ai_rd_|machine_learning_local\/|debug_small_libs\/)/, "").replace(/\/main$/, "");
const shortModel = (m: string | null) => (m ?? "").replace(/^.*\//, "").replace(/-\d{8}$/, "");

function idsByKey(array: ArrayData): Map<string, EntityID> {
  return new Map(rowsOf(array).flatMap(({ key, value }) => (isEntityID(value) ? [[keyOf(key), value] as const] : [])));
}

export function Discrepancies({ discrepancies, runs, judgments, actionJudgments }: {
  discrepancies: EntityID<ArrayData>; runs: EntityID<ArrayData>; judgments: EntityID<ArrayData>; actionJudgments: EntityID<ArrayData>;
}) {
  const runIds = idsByKey(unwrap(useEntity(runs)));
  const full = idsByKey(unwrap(useEntity(judgments)));
  const actions = idsByKey(unwrap(useEntity(actionJudgments)));
  const items: Item[] = rowsOf(unwrap(useEntity(discrepancies))).map(({ key, value }) => {
    const name = keyOf(key);
    return { name, task: String(key[0]), d: value as unknown as Discrepancy, run: runIds.get(name), full: full.get(name), actions: actions.get(name) };
  });
  const counts = new Map(KINDS.map((k) => [k, items.filter((i) => i.d.kind === k).length]));
  const [shown, setShown] = useState<ReadonlySet<Kind>>(new Set(KINDS));
  const toggle = (k: Kind) => setShown((s) => {
    const next = new Set(s);
    if (next.has(k)) next.delete(k); else next.add(k);
    return next;
  });
  const visible = items
    .filter((i) => shown.has(i.d.kind))
    .sort((a, b) => KINDS.indexOf(a.d.kind) - KINDS.indexOf(b.d.kind) || a.name.localeCompare(b.name));

  return (
    <>
      <h3 className={layout.subtitle}>discrepancies</h3>
      <p className={layout.muted}>
        Runs where MALT and the two judges do not all agree ({items.length} of the audit's runs). Each card shows the rules a judge
        found broken with its evidence; a quote chip opens the run at the cited message.
      </p>
      <div className={styles.filters}>
        {KINDS.map((k) => (
          <button key={k} type="button" className={styles.filter} aria-pressed={shown.has(k)} onClick={() => toggle(k)} title={KIND_NOTE[k]}>
            {k} <span className={styles.count}>{counts.get(k) ?? 0}</span>
          </button>
        ))}
      </div>
      <div className={styles.list}>
        {visible.map((i) => <Card key={i.name} item={i} />)}
        {visible.length === 0 && <p className={layout.muted}>No runs of the kinds selected.</p>}
      </div>
    </>
  );
}

function Card({ item }: { item: Item }) {
  const { d } = item;
  return (
    <section className={styles.card} data-kind={d.kind}>
      <header className={styles.head}>
        <span className={styles.kind} data-kind={d.kind}>{d.kind}</span>
        <span className={styles.task}>{shortTask(item.task)}</span>
        {item.run ? <EntityLink id={item.run} /> : <span>{item.name}</span>}
        <span className={styles.meta}>{shortModel(d.model)}</span>
        <span className={styles.meta}>MALT labels: {d.malt_labels.join(", ") || "none"}</span>
      </header>
      <div className={styles.calls}>
        <Call label="MALT" flagged={d.malt} />
        <Call label="full judge" flagged={d.full.broke} judged={d.full.judged} />
        <Call label="actions-only judge" flagged={d.actions.broke} judged={d.actions.judged} />
      </div>
      <div className={styles.sides}>
        <Side label="full judge" side={d.full} judgment={item.full} />
        <Side label="actions-only judge" side={d.actions} judgment={item.actions} />
      </div>
    </section>
  );
}

function Call({ label, flagged, judged = true }: { label: string; flagged: boolean; judged?: boolean }) {
  return (
    <div className={styles.call} data-flagged={judged && flagged}>
      <span className={styles.callLabel}>{label}</span>
      <span className={styles.callValue}>{!judged ? "no answer" : flagged ? "broke a rule" : "no break"}</span>
    </div>
  );
}

function Side({ label, side, judgment }: { label: string; side: JudgeSide; judgment: EntityID | undefined }) {
  return (
    <div className={styles.side}>
      <div className={styles.sideHead}>
        <span className={styles.sideLabel}>{label}</span>
        {judgment && <EntityLink id={judgment} />}
      </div>
      {judgment ? <SideBody side={side} judgment={judgment} /> : <p className={layout.muted}>No judgment.</p>}
    </div>
  );
}

function SideBody({ side, judgment }: { side: JudgeSide; judgment: EntityID }) {
  const j = judgmentOf(unwrap(useEntity(judgment)));
  if (j === null || j.output === null) return <p className={layout.muted}>The judge gave no answer.</p>;
  const out = j.output as Record<string, unknown>;
  const verdicts = (out["verdicts"] ?? []) as readonly Record<string, unknown>[];
  return (
    <>
      {side.broken.map((r) => (
        <div key={r.rule_id} className={styles.rule}>
          <div className={styles.ruleHead}>
            <span className={styles.ruleId}>{r.rule_id}</span>
            <span className={styles.verdict} data-verdict={r.verdict}>{r.verdict}</span>
            {r.ambiguous && <span className={styles.tag}>ambiguous</span>}
            <span className={styles.tag}>{r.confidence} confidence</span>
            {r.disclosure && <span className={styles.tag}>{r.disclosure}</span>}
          </div>
          <div className={styles.ruleText}>{r.rule}</div>
          {r.quote && <blockquote className={styles.quote}>{r.quote}</blockquote>}
          <Field j={j} path={["verdicts", r.index, "evidence"]} text={String(verdicts[r.index]?.["evidence"] ?? "")} />
        </div>
      ))}
      {side.broken.length === 0 && (
        <Section title="discussion"><Field j={j} path={["discussion"]} text={String(out["discussion"] ?? "")} /></Section>
      )}
      <Optional title="circumvention" j={j} field="circumvention" />
      <Optional title="rules feedback" j={j} field="rules_feedback" />
    </>
  );
}

function Optional({ title, j, field }: { title: string; j: JudgmentData; field: string }) {
  const text = String((j.output as Record<string, unknown>)[field] ?? "").trim();
  if (!text) return null;
  return <Section title={title}><Field j={j} path={[field]} text={text} /></Section>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className={styles.section}>
      <summary>{title}</summary>
      {children}
    </details>
  );
}

const samePath = (a: readonly (string | number)[], b: readonly (string | number)[]) => a.length === b.length && a.every((p, i) => p === b[i]);

/** The index in `text` of code point number `offset` (citation offsets count code points). */
function indexOfCodePoint(text: string, offset: number): number {
  let index = 0;
  for (let count = 0; count < offset && index < text.length; count++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return index;
}

/** A field of the judge's answer as prose, each citation in it a chip naming the message it cites. */
function Field({ j, path, text }: { j: JudgmentData; path: readonly (string | number)[]; text: string }) {
  const cites = j.citations.filter((c) => samePath(c.path_to_output_field, path)).sort((a, b) => a.start_in_field - b.start_in_field);
  const pieces: ReactNode[] = [];
  let at = 0;
  cites.forEach((c, n) => {
    pieces.push(text.slice(at, indexOfCodePoint(text, c.start_in_field)));
    pieces.push(<Cite key={n} c={c} />);
    at = indexOfCodePoint(text, c.end_in_field);
  });
  pieces.push(text.slice(at));
  return <div className={styles.prose}>{pieces}</div>;
}

/** A citation chip: opens the run the judge read at the cited message (the reader's `selector`). */
function Cite({ c }: { c: CitationData }) {
  const links = useEntityLinks();
  const m = c.address_in_entity ? /^(?:T(\d+))?B(\d+)L(\d+)(?:-L?(\d+))?$/.exec(c.address_in_entity) : null;
  const params = m
    ? { selector: { ...(m[1] ? { transcript: Number(m[1]) } : {}), message: Number(m[2]), lines: [Number(m[3]), Number(m[4] ?? m[3])] } }
    : undefined;
  const label = m ? `msg ${m[2]}` : (c.address_in_entity ?? "source");
  return (
    <button type="button" className={styles.cite} aria-pressed={links.isOpen(c.entity_id, params)} title={`Open ${c.address_in_entity ?? ""}`}
      onClick={() => links.open(c.entity_id, params)}>
      {label}
    </button>
  );
}
