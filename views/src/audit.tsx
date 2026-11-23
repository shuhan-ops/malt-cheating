// The stated-rules audit (src/malt_fxtr/audit.py), read from its `Audit` report: how each judge
// agrees with MALT's `ignores_task_instructions` label, then a table of every run with both
// judges' verdicts and the disclosure level, linking to the run (in the reader) and to each
// judgment (whose citations open the cited message).

import {
  EntityLink,
  isEntityID,
  rowsOf,
  scalarEntityId,
  unwrap,
  useEntity,
  useInvocation,
  type ArrayData,
  type DataMap,
  type EntityData,
  type EntityID,
  type EntityPanelProps,
  type EntityViewProps,
  type InvocationPanelProps,
} from "fxtr-view";
import { Discrepancies } from "./discrepancies";
import { DataTable, InvocationHead, InvocationOutcome, Kind, Stat, Stats, formatNumber, layout, type TableColumn } from "fxtr-view-kit";

export const AUDIT = "local.malt_fxtr.Audit.v1";
export const AUDIT_WORKFLOW = "malt.rules_audit";

interface Agreement {
  readonly judged: number;
  readonly both: number;
  readonly ours_only: number;
  readonly malt_only: number;
  readonly neither: number;
}

interface Summary {
  readonly runs: number;
  readonly full: Agreement;
  readonly actions: Agreement;
  readonly judges_disagree: readonly string[];
  readonly disclosure: { readonly hidden: number; readonly mentioned: number; readonly flagged: number };
}

interface AuditData extends EntityData {
  readonly $type: typeof AUDIT;
  readonly extract_model: string;
  readonly judge_model: string;
  readonly summary: DataMap;
  readonly rows: EntityID<ArrayData>;
  readonly rules: EntityID<ArrayData>;
  // Jobs before task texts were grouped have neither of these.
  readonly task_texts?: EntityID<ArrayData>; // [text]: {id, text, runs}
  readonly extractions?: EntityID<ArrayData>; // [text]: Judgment
  readonly judgments: EntityID<ArrayData>;
  readonly action_judgments: EntityID<ArrayData>;
  readonly runs: EntityID<ArrayData>;
  readonly action_runs: EntityID<ArrayData>;
  // Jobs before the discrepancy stage have none.
  readonly discrepancies?: EntityID<ArrayData> | null;
}

interface JudgeRow {
  readonly judged: boolean;
  readonly broke_stated_rule: boolean | null;
  readonly violated: number;
  readonly attempted: number;
  readonly ambiguous: number;
  readonly worst_disclosure: string | null;
}

interface AuditRow {
  readonly model: string | null;
  readonly malt_labels: readonly string[];
  readonly malt_ignores_instructions: boolean;
  readonly full: JudgeRow;
  readonly actions: JudgeRow;
}

interface Row {
  readonly name: string;
  readonly task: string;
  readonly run: string;
  readonly row: AuditRow;
  readonly runId: EntityID | undefined;
  readonly rules: EntityID | undefined;
  readonly full: EntityID | undefined;
  readonly actions: EntityID | undefined;
}

const keyOf = (key: readonly unknown[]) => `${String(key[0])}/${String(key[1])}`;

function idsByKey(array: ArrayData): Map<string, EntityID> {
  return new Map(rowsOf(array).flatMap(({ key, value }) => (isEntityID(value) ? [[keyOf(key), value] as const] : [])));
}

function verdict(j: JudgeRow): string {
  if (!j.judged) return "no answer";
  if (!j.broke_stated_rule) return "–";
  const parts = [j.violated ? `${j.violated} violated` : "", j.attempted ? `${j.attempted} attempted` : ""].filter(Boolean);
  return parts.join(", ") + (j.ambiguous ? ` (${j.ambiguous} ambiguous)` : "");
}

function outcome(row: AuditRow, j: JudgeRow): string {
  const ours = Boolean(j.broke_stated_rule);
  const theirs = row.malt_ignores_instructions;
  return ours && theirs ? "both" : ours ? "ours only" : theirs ? "MALT only" : "neither";
}

function AgreementStat({ label, a }: { label: string; a: Agreement }) {
  const p = a.both + a.ours_only ? a.both / (a.both + a.ours_only) : null;
  const r = a.both + a.malt_only ? a.both / (a.both + a.malt_only) : null;
  return (
    <Stat
      label={label}
      value={`${a.both} / ${a.both + a.malt_only}`}
      note={`MALT positives caught · ${a.ours_only} ours only · precision ${p == null ? "–" : p.toFixed(2)}, recall ${r == null ? "–" : r.toFixed(2)}`}
    />
  );
}

export function AuditView({ id }: { id: EntityID<AuditData> }) {
  const audit = unwrap(useEntity(id));
  const rowsArr = unwrap(useEntity(audit.rows));
  const runs = idsByKey(unwrap(useEntity(audit.runs)));
  const rules = idsByKey(unwrap(useEntity(audit.rules)));
  const full = idsByKey(unwrap(useEntity(audit.judgments)));
  const actions = idsByKey(unwrap(useEntity(audit.action_judgments)));
  const s = audit.summary as unknown as Summary;

  const rows: Row[] = rowsOf(rowsArr).map(({ key, value }) => {
    const name = keyOf(key);
    return {
      name, task: String(key[0]), run: String(key[1]), row: value as unknown as AuditRow,
      runId: runs.get(name), rules: rules.get(name), full: full.get(name), actions: actions.get(name),
    };
  });

  // Few columns, each one link or one word, so the table reads in a narrow pane; the task's family
  // prefix (ai_rd_, machine_learning_local/) is dropped, and the model's date suffix.
  const shortTask = (t: string) => t.replace(/^(ai_rd_|machine_learning_local\/|debug_small_libs\/)/, "").replace(/\/main$/, "");
  const shortModel = (m: string | null) => (m ?? "").replace(/^.*\//, "").replace(/-\d{8}$/, "");
  const columns: TableColumn<Row>[] = [
    { key: "task", label: "task", width: 110, value: (r) => shortTask(r.task), filterable: true },
    { key: "run", label: "run", width: 150, value: (r) => r.run, cell: (r) => (r.runId ? <EntityLink id={r.runId} /> : r.run) },
    { key: "model", label: "model", width: 120, value: (r) => shortModel(r.row.model), filterable: true },
    { key: "full_vs", label: "vs MALT", width: 84, value: (r) => outcome(r.row, r.row.full), filterable: true },
    { key: "full", label: "full judge", width: 150, value: (r) => verdict(r.row.full), cell: (r) => (r.full ? <EntityLink id={r.full} /> : verdict(r.row.full)) },
    { key: "actions", label: "actions-only", width: 150, value: (r) => verdict(r.row.actions), cell: (r) => (r.actions ? <EntityLink id={r.actions} /> : verdict(r.row.actions)) },
    { key: "disclosure", label: "disclosure", width: 90, value: (r) => r.row.full.worst_disclosure ?? "", filterable: true },
  ];

  return (
    <>
      <Stats>
        <AgreementStat label="full judge" a={s.full} />
        <AgreementStat label="actions-only judge" a={s.actions} />
        <Stat label="judges disagree" value={formatNumber(s.judges_disagree.length)} note={`of ${formatNumber(s.runs)} runs`} />
        <Stat label="disclosure (full)" value={`${s.disclosure.hidden} hidden`} note={`${s.disclosure.mentioned} mentioned · ${s.disclosure.flagged} flagged`} />
      </Stats>
      <p className={layout.muted}>
        Rules extracted by {audit.extract_model}; both judges are {audit.judge_model}. Open a run to read it, or a judgment to
        see each rule's verdict with citations into the run.
      </p>
      {audit.discrepancies && (
        <Discrepancies discrepancies={audit.discrepancies} runs={audit.runs} judgments={audit.judgments} actionJudgments={audit.action_judgments} />
      )}
      {audit.task_texts && audit.extractions && <TaskTexts texts={audit.task_texts} extractions={audit.extractions} />}
      <h3 className={layout.subtitle}>runs</h3>
      <DataTable rows={rows} columns={columns} rowKey={(r) => r.name} initialSort={{ key: "full_vs", direction: "desc" }} />
    </>
  );
}

export function AuditPanel({ id }: EntityPanelProps) {
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>stated-rules audit</Kind></div>
      <AuditView id={id as EntityID<AuditData>} />
    </div>
  );
}

export function AuditInlineLink({ id }: EntityViewProps) {
  const audit = unwrap(useEntity(id as EntityID<AuditData>));
  return <span className={layout.link}><Kind>stated-rules audit</Kind> {(audit.summary as unknown as Summary).runs} runs</span>;
}

export function AuditOverview({ invocations }: InvocationPanelProps) {
  const [run] = invocations;
  if (invocations.length !== 1 || run === undefined)
    throw new Error(`a results overview is handed its run's one invocation, not ${invocations.length}`);
  const invocation = unwrap(useInvocation(run.ref));
  return (
    <div className={layout.panel}>
      <InvocationHead kind="workflow" name={invocation.name} />
      {invocation.output ? <Overview output={invocation.output} /> : <InvocationOutcome invocation={invocation} />}
    </div>
  );
}

function Overview({ output }: { output: EntityID<ArrayData> }) {
  return <AuditView id={scalarEntityId(unwrap(useEntity(output))) as EntityID<AuditData>} />;
}

interface TaskTextRow {
  readonly id: string;
  readonly text: string;
  readonly runs: number;
  readonly extraction: EntityID | undefined;
}

/** The review queue: each distinct task text, most runs first, with the rules extracted from it. */
function TaskTexts({ texts, extractions }: { texts: EntityID<ArrayData>; extractions: EntityID<ArrayData> }) {
  const ex = new Map(rowsOf(unwrap(useEntity(extractions))).flatMap(({ key, value }) => (isEntityID(value) ? [[String(key[0]), value] as const] : [])));
  const rows: TaskTextRow[] = rowsOf(unwrap(useEntity(texts))).map(({ key, value }) => {
    const v = value as unknown as { text: string; runs: number };
    return { id: String(key[0]), text: v.text, runs: v.runs, extraction: ex.get(String(key[0])) };
  });
  const columns: TableColumn<TaskTextRow>[] = [
    { key: "runs", label: "runs", numeric: true, width: 60, value: (r) => r.runs },
    { key: "rules", label: "extracted rules", width: 170, value: (r) => r.id, cell: (r) => (r.extraction ? <EntityLink id={r.extraction} /> : r.id) },
    { key: "text", label: "task text", value: (r) => r.text.replace(/\s+/g, " ").slice(0, 200) },
  ];
  return (
    <>
      <h3 className={layout.subtitle}>task texts to review</h3>
      <p className={layout.muted}>Runs grouped by task text (one text containing another counts as the same). Rules are extracted once per text; review the most used first.</p>
      <DataTable rows={rows} columns={columns} rowKey={(r) => r.id} initialSort={{ key: "runs", direction: "desc" }} />
    </>
  );
}
