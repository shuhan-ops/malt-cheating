// A readable view of a MALT agent run (a behaviors `AgentRun` from load.py): the task, then each
// message with its index (the number a judge's citation `T0B<index>` points to), markdown rendered,
// reasoning folded under "Thinking", each tool call as a code block with its result folded when
// long, and the scaffold's notices (token usage, advisor prompts) dimmed to one line. It reads the
// same `selector` param as behaviors' conversation view, so a citation opened from a judgment
// scrolls to and marks the cited message here too.

import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { EntityPanel, declareView, unwrap, useEntity, type DataMap, type EntityID, type ParamValue, type ViewProps } from "fxtr-view";
import { AGENT_RUN, type ContentData, type MessageData, type TranscriptGroupData } from "fxtr-view-behaviors";
import styles from "./reader.module.css";

export const RUN_READER = declareView(EntityPanel, AGENT_RUN, "malt-fxtr.reader", { params: ["selector"] });

const NOTICE = /^(So far in this attempt|\d+ of \d+ tokens used|You have used|\[system prompt for a new model call\])/;

function selectedMessage(value: ParamValue | undefined): number | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const m = (value as { readonly [k: string]: ParamValue })["message"];
  return typeof m === "number" && Number.isInteger(m) ? m : null;
}

function textOf(m: MessageData): string {
  return m.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("\n");
}

function Markdown({ text }: { text: string }) {
  return (
    <div className={styles.markdown}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

/** Long text shown to `limit` lines, with a button to show the rest. */
function Folded({ text, limit, className, empty = "(no output)" }: { text: string; limit: number; className: string | undefined; empty?: string }) {
  const [open, setOpen] = useState(false);
  if (!text.trim()) return <pre className={className} data-empty="true">{empty}</pre>;
  const lines = text.split("\n");
  const long = lines.length > limit;
  return (
    <>
      <pre className={className}>{open || !long ? text : lines.slice(0, limit).join("\n")}</pre>
      {long && (
        <button type="button" className={styles.more} onClick={() => setOpen(!open)}>
          {open ? "show less" : `show all ${lines.length} lines`}
        </button>
      )}
    </>
  );
}

function ToolCall({ name, input }: { name: string; input: DataMap | null }) {
  const args = (input ?? {}) as { readonly [k: string]: unknown };
  const main = ["command", "code", "cmd", "script"].find((k) => typeof args[k] === "string");
  const rest = Object.fromEntries(Object.entries(args).filter(([k]) => k !== main));
  return (
    <div className={styles.call}>
      <div className={styles.callHead}>
        ▸ <b>{name}</b>
        {Object.keys(rest).length > 0 && <> {JSON.stringify(rest)}</>}
      </div>
      {main && <Folded text={String(args[main])} limit={40} className={styles.code} empty="(empty)" />}
    </div>
  );
}

function Block({ block, showThinking }: { block: ContentData; showThinking: boolean }) {
  if (block.type === "text") return <Markdown text={block.text} />;
  if (block.type === "reasoning") {
    const words = block.text.split(/\s+/).filter(Boolean).length;
    return (
      <details className={styles.thinking} open={showThinking}>
        <summary>Thinking · {words} words</summary>
        <Markdown text={block.text} />
      </details>
    );
  }
  return <ToolCall name={block.name} input={block.input} />;
}

type Kind = "system" | "task" | "notice" | "advisor" | "tool_output" | "user" | "assistant" | "tool";

function kindOf(m: MessageData, index: number, acted: boolean): Kind {
  if (m.role === "system") return "system";
  if (m.role === "assistant") return "assistant";
  if (m.role === "tool") return "tool";
  const text = textOf(m);
  if (!acted && index <= 3 && text.length > 150) return "task";
  if (text.startsWith("[tool output")) return "tool_output";
  if (text.startsWith("<advisor>")) return "advisor";
  if (NOTICE.test(text.trim()) || text.trim().length < 140) return "notice";
  return "user";
}

const ROLE_LABEL: Record<Kind, string> = {
  system: "system prompt", task: "task", notice: "scaffold", advisor: "advisor", tool_output: "tool output",
  user: "user", assistant: "agent", tool: "result",
};

function Message({ m, kind, showThinking }: { m: MessageData; kind: Kind; showThinking: boolean }) {
  const text = textOf(m);
  switch (kind) {
    case "system":
      return (
        <details className={styles.fold}>
          <summary>System prompt · {text.length.toLocaleString()} characters</summary>
          <Markdown text={text} />
        </details>
      );
    case "task":
      return <Markdown text={text} />;
    case "notice":
      return <div className={styles.notice}>{text.trim().replace(/\s+/g, " ")}</div>;
    case "advisor":
      return (
        <details className={styles.fold} open={showThinking}>
          <summary>Advisor's suggestion</summary>
          <Markdown text={text.replace(/<\/?advisor>/g, "")} />
        </details>
      );
    case "tool":
    case "tool_output":
      return <Folded text={text.replace(/^\[tool output: [^\]]*\]\n?/, "")} limit={20} className={styles.output} />;
    default:
      return <>{m.content.map((c, i) => <Block key={i} block={c} showThinking={showThinking} />)}</>;
  }
}

function Chip({ children, tone }: { children: ReactNode; tone?: "flag" | undefined }) {
  return <span className={styles.chip} data-tone={tone}>{children}</span>;
}

export function RunReader({ id, params }: ViewProps<typeof RUN_READER>) {
  const run = unwrap(useEntity(id as EntityID<TranscriptGroupData>));
  const [showThinking, setShowThinking] = useState(false);
  const [showNotices, setShowNotices] = useState(true);
  const selected = selectedMessage(params?.selector);
  const selectedRef = useRef<HTMLDivElement | null>(null);
  // A pane that has just opened is not laid out yet when the effect runs, so scroll again once it is.
  useEffect(() => {
    const scroll = () => selectedRef.current?.scrollIntoView({ block: "center" });
    scroll();
    const timers = [100, 400].map((ms) => setTimeout(scroll, ms));
    return () => timers.forEach(clearTimeout);
  }, [id, selected]);

  const meta = run.metadata as { readonly [k: string]: unknown };
  const labels = Array.isArray(meta["labels"]) ? (meta["labels"] as string[]) : [];
  const messages = run.transcripts[0]?.messages ?? [];
  let acted = false;
  const kinds = messages.map((m, i) => {
    const k = kindOf(m, i, acted);
    if (m.role === "assistant") acted = true;
    return k;
  });
  const calls = messages.reduce((n, m) => n + m.content.filter((c) => c.type === "tool_use").length, 0);

  return (
    <div className={styles.reader}>
      <div className={styles.head}>
        <span className={styles.title}>Run {run.name}</span>
        <span className={styles.meta}>{String(meta["task_id"] ?? "")} · {String(meta["model"] ?? "")}</span>
        <span className={styles.meta}>{messages.length} messages · {calls} tool calls</span>
      </div>
      <div className={styles.chips}>
        {labels.map((l) => <Chip key={l} tone={l === "normal" ? undefined : "flag"}>{l}</Chip>)}
        {meta["run_source"] ? <Chip>{String(meta["run_source"])}</Chip> : null}
        {meta["manually_reviewed"] ? <Chip>reviewed by hand</Chip> : <Chip>not reviewed</Chip>}
      </div>
      <div className={styles.controls}>
        <label><input type="checkbox" checked={showThinking} onChange={(e) => setShowThinking(e.target.checked)} /> expand thinking</label>
        <label><input type="checkbox" checked={showNotices} onChange={(e) => setShowNotices(e.target.checked)} /> show scaffold notices</label>
      </div>
      {messages.map((m, i) => {
        const kind = kinds[i] as Kind;
        const isSelected = i === selected;
        if (kind === "notice" && !showNotices && !isSelected) return null;
        return (
          <div key={i} className={styles.msg} data-selected={isSelected} ref={isSelected ? selectedRef : undefined}>
            <div className={styles.index} title="message index: a judge cites it as T0B<index>">{i}</div>
            <div className={styles.body}>
              {kind !== "notice" && <div className={styles.role} data-role={m.role}>{ROLE_LABEL[kind]}</div>}
              <Message m={m} kind={kind} showThinking={showThinking} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
