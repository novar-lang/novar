import init, {
  check, functions, run, hover,
  repl_eval, repl_load, repl_type, repl_vars, repl_defs, repl_reset, repl_is_complete, repl_complete,
} from "./pkg/telvax_wasm.js";
import { EXAMPLES } from "./examples.js";

const $ = (id) => document.getElementById(id);
const code = $("code"), highlight = $("highlight"), output = $("output"), problems = $("problems");
const status = $("status"), fnSelect = $("function"), argsBox = $("args"), exampleSelect = $("examples");

let ready = false;
let currentExample = EXAMPLES[0];

// ---- syntax highlighting (a light tokenizer; the compiler is the source of truth) ------------
const TOKEN = /(#.*)|(f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')|\b(fn|let|var|type|enum|trait|extend|import|from|return|if|elif|else|while|for|in|is|match|case|break|continue|pass|and|or|not|True|False|None)\b|\b(Int|Float|Text|Bool|i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|List|Map|Vector|Tensor|Table|Result|[A-Z]\w*)\b|\b(\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)\b|(\|>|->|=>|\.\.\.|[-+*\/%=<>!|?]+)|(@\w+)|([A-Za-z_]\w*)(?=\s*\()/g;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function render(src) {
  let html = "", last = 0;
  for (const m of src.matchAll(TOKEN)) {
    html += esc(src.slice(last, m.index));
    const cls = m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "t" : m[5] ? "n" : m[6] ? "o" : m[7] ? "f" : "f";
    html += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  highlight.innerHTML = html + esc(src.slice(last)) + "\n";
}

function syncScroll() {
  highlight.scrollTop = code.scrollTop;
  highlight.scrollLeft = code.scrollLeft;
}

// ---- checking -----------------------------------------------------------------------------------
let timer;
function scheduleCheck() {
  clearTimeout(timer);
  timer = setTimeout(analyse, 200);
}

function analyse() {
  render(code.value);
  if (!ready) return;
  const r = JSON.parse(check(code.value));
  problems.innerHTML = "";
  $("problem-count").textContent = r.diagnostics.length ? `(${r.diagnostics.length})` : "";
  if (!r.diagnostics.length) {
    problems.innerHTML = '<li class="ok">No problems</li>';
  }
  for (const d of r.diagnostics) {
    const li = document.createElement("li");
    li.className = d.severity;
    li.textContent = `${d.startLine + 1}:${d.startCol + 1}  ${d.message}`;
    li.onclick = () => jumpTo(d.startLine, d.startCol);
    problems.append(li);
  }
  refreshFunctions();
  updateStatus();
}

function jumpTo(line, col) {
  const lines = code.value.split("\n");
  let offset = 0;
  for (let i = 0; i < line; i++) offset += lines[i].length + 1;
  code.focus();
  code.setSelectionRange(offset + col, offset + col);
}

function caretPosition() {
  const before = code.value.slice(0, code.selectionStart).split("\n");
  return { line: before.length - 1, character: before[before.length - 1].length };
}

function updateStatus() {
  if (!ready) return;
  const { line, character } = caretPosition();
  const info = hover(code.value, line, character);
  status.textContent = info ? info.replace(/```telvax\n?|```/g, "").trim() : `Ln ${line + 1}, Col ${character + 1}`;
}

// ---- running ------------------------------------------------------------------------------------
function refreshFunctions() {
  const fns = JSON.parse(functions(code.value));
  const wanted = fnSelect.value || currentExample.run;
  fnSelect.innerHTML = "";
  for (const f of fns) {
    const o = document.createElement("option");
    o.value = f.name;
    o.textContent = `${f.name}(${f.params.map((p) => p.name).join(", ")})`;
    o.dataset.params = JSON.stringify(f.params);
    fnSelect.append(o);
  }
  if (fns.some((f) => f.name === wanted)) fnSelect.value = wanted;
  buildArgInputs(false);
}

function buildArgInputs(useExample) {
  const opt = fnSelect.selectedOptions[0];
  const params = opt ? JSON.parse(opt.dataset.params) : [];
  const previous = [...argsBox.querySelectorAll("input")].map((i) => i.value);
  argsBox.innerHTML = "";
  params.forEach((p, i) => {
    const input = document.createElement("input");
    input.title = `${p.name}: ${p.ty} (JSON)`;
    input.placeholder = `${p.name}: ${p.ty}`;
    const fromExample = useExample && currentExample.args && fnSelect.value === currentExample.run;
    input.value = fromExample ? JSON.stringify(currentExample.args[i]) : (previous[i] ?? "");
    argsBox.append(input);
  });
}

function runProgram() {
  if (!ready) return;
  const args = [...argsBox.querySelectorAll("input")].map((i) => i.value.trim());
  const json = "[" + args.join(",") + "]";
  const started = performance.now();
  const r = JSON.parse(run(code.value, fnSelect.value, json));
  const ms = (performance.now() - started).toFixed(1);
  output.innerHTML = "";
  const add = (cls, text) => {
    const span = document.createElement("span");
    span.className = cls;
    span.textContent = text;
    output.append(span);
  };
  if (r.output) add("", r.output.endsWith("\n") ? r.output : r.output + "\n");
  if (r.ok) {
    if (r.result !== null) add("ok", `=> ${JSON.stringify(r.result, null, 2)}\n`);
    add("muted", `finished in ${ms} ms`);
  } else {
    add("error", `error: ${r.error}\n`);
    if (r.diagnostics?.length) add("muted", "see Problems");
  }
}

// ---- sharing and examples ---------------------------------------------------------------------
const encode = (s) => btoa(unescape(encodeURIComponent(s)));
const decode = (s) => decodeURIComponent(escape(atob(s)));

function loadExample(ex) {
  currentExample = ex;
  code.value = ex.source;
  fnSelect.value = ex.run;
  output.textContent = "";
  analyse();
  fnSelect.value = ex.run;
  buildArgInputs(true);
}

EXAMPLES.forEach((ex, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = ex.name;
  exampleSelect.append(o);
});

exampleSelect.onchange = () => loadExample(EXAMPLES[Number(exampleSelect.value)]);
fnSelect.onchange = () => buildArgInputs(true);
$("run").onclick = runProgram;
$("share").onclick = async () => {
  const url = `${location.origin}${location.pathname}#${encode(code.value)}`;
  history.replaceState(null, "", url);
  try { await navigator.clipboard.writeText(url); status.textContent = "Link copied to the clipboard"; }
  catch { status.textContent = "Link is in the address bar"; }
};
code.addEventListener("input", scheduleCheck);
code.addEventListener("scroll", syncScroll);
for (const ev of ["keyup", "click"]) code.addEventListener(ev, updateStatus);
code.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runProgram(); }
  if (e.key === "Tab") {
    e.preventDefault();
    const { selectionStart: s, selectionEnd: end } = code;
    code.setRangeText("    ", s, end, "end");
    scheduleCheck();
  }
});

// ---- the console ---------------------------------------------------------------------------------
const consoleLog = $("console-log"), consoleInput = $("console-input"), consolePrompt = $("console-prompt");
const CONSOLE_HELP = `Type a definition (fn, type, enum, trait, extend, import), a variable (let, var), a statement
or an expression. A block ends with an empty line. Names last until :reset.
  :type <expression>   the type of an expression, without running it
  :vars                the variables, with their types and values
  :defs                the functions, types and the like defined so far
  :load                enter the definitions and variables of the program above
  :reset               forget everything
  :clear               clear the screen (Ctrl+L)
Tab completes names, fields and methods; the arrow keys go through earlier lines.`;
const encoder = new TextEncoder(), decoder = new TextDecoder();
let pending = "";
const lines = [];
let lineAt = 0;

function say(cls, text) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = text;
  consoleLog.append(div);
  const box = $("console");
  box.scrollTop = box.scrollHeight;
}

/** What an input did: what it printed, its value, what it defined, what went wrong. */
function showOutcome(r) {
  if (r.output) say("", r.output.replace(/\n$/, ""));
  if (r.value) say("ok", r.value.text);
  for (const note of r.notes ?? []) say("muted", `note: ${note}`);
  if (r.error) say("error", r.error);
}

function consoleCommand(line) {
  const [word, ...rest] = line.trim().split(/\s+/);
  const arg = rest.join(" ");
  switch (word) {
    case ":help": case ":h": case ":?": say("muted", CONSOLE_HELP); break;
    case ":clear": consoleLog.innerHTML = ""; break;
    case ":reset": repl_reset(); say("muted", "(everything forgotten)"); break;
    case ":load": loadProgram(); break;
    case ":vars": {
      const vars = JSON.parse(repl_vars());
      if (!vars.length) say("muted", "(no variables)");
      for (const v of vars) say("", `${v.name}: ${v.ty} = ${v.value}`);
      break;
    }
    case ":defs": {
      const defs = JSON.parse(repl_defs());
      if (!defs.length) say("muted", "(nothing defined)");
      for (const d of defs) say("", d);
      break;
    }
    case ":type": case ":t": {
      if (!arg) { say("error", "error: :type takes an expression, as in `:type 1 + 2`"); break; }
      const r = JSON.parse(repl_type(arg));
      say(r.ok ? "ok" : "error", r.ok ? r.type : r.error);
      break;
    }
    default: say("error", `error: no command \`${word}\`; :help lists them`);
  }
}

function loadProgram() {
  if (!ready) return;
  const r = JSON.parse(repl_load(code.value));
  if (r.error) showOutcome(r);
  else say("muted", r.defined.length ? `entered ${r.defined.join(", ")}` : "(nothing to enter)");
}

/** The spaces to start the next line with: those of the line before, and one level more after a `:`. */
function nextIndent(text) {
  const last = text.split("\n").filter((l) => l.trim()).pop();
  if (!last) return "";
  const spaces = last.length - last.trimStart().length;
  return " ".repeat(spaces + (last.trimEnd().endsWith(":") ? 4 : 0));
}

function submitLine() {
  if (!ready) { say("muted", "The compiler is still loading…"); return; }
  const line = consoleInput.value;
  if (!pending && !line.trim()) return;
  say("echo", `${pending ? "..." : ">>>"} ${line}`);
  if (line.trim()) { lines.push(line); lineAt = lines.length; }
  consoleInput.value = "";
  if (!pending && line.trimStart().startsWith(":")) { consoleCommand(line); return; }
  pending = pending ? `${pending}\n${line}` : line;
  // An empty line ends a block; otherwise the input ends when it is complete.
  if (!line.trim() || repl_is_complete(pending)) {
    showOutcome(JSON.parse(repl_eval(pending)));
    pending = "";
    consolePrompt.textContent = ">>>";
  } else {
    consolePrompt.textContent = "...";
    consoleInput.value = nextIndent(pending);
  }
}

function commonPrefix(words) {
  let prefix = words[0] ?? "";
  for (const w of words) while (!w.startsWith(prefix)) prefix = prefix.slice(0, -1);
  return prefix;
}

function completeInput() {
  const line = consoleInput.value, at = consoleInput.selectionStart ?? line.length;
  if (!line.slice(0, at).trim()) { consoleInput.setRangeText("    ", at, at, "end"); return; }
  if (!ready) return;
  // The engine counts bytes of UTF-8; the page counts UTF-16 units.
  const cursor = encoder.encode(line.slice(0, at)).length;
  const r = JSON.parse(repl_complete(line, cursor));
  const start = decoder.decode(encoder.encode(line).slice(0, r.start)).length;
  const labels = [...new Set(r.items.map((i) => i.label))];
  if (!labels.length) return;
  const common = commonPrefix(labels);
  if (common.length > at - start) consoleInput.setRangeText(common, start, at, "end");
  if (labels.length > 1) say("muted", labels.slice(0, 40).join("  "));
}

consoleInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitLine(); }
  else if (e.key === "Tab") { e.preventDefault(); completeInput(); }
  else if (e.key === "ArrowUp" && lineAt > 0) { e.preventDefault(); consoleInput.value = lines[--lineAt]; }
  else if (e.key === "ArrowDown" && lineAt < lines.length) { e.preventDefault(); consoleInput.value = lines[++lineAt] ?? ""; }
  else if (e.key === "l" && e.ctrlKey) { e.preventDefault(); consoleLog.innerHTML = ""; }
});
$("console-load").onclick = loadProgram;
$("console-reset").onclick = () => { if (ready) { repl_reset(); pending = ""; consolePrompt.textContent = ">>>"; say("muted", "(everything forgotten)"); } };
$("console-clear").onclick = () => { consoleLog.innerHTML = ""; };
say("muted", "Console: type a definition, a variable or an expression. :help lists the commands.");

render(EXAMPLES[0].source);
code.value = EXAMPLES[0].source;
try {
  await init();
  ready = true;
  if (location.hash.length > 1) {
    try { code.value = decode(location.hash.slice(1)); currentExample = { run: "main" }; } catch { /* ignore a bad link */ }
  }
  analyse();
  if (!location.hash.length > 1) buildArgInputs(true);
  status.textContent = "Ready. Press Ctrl+Enter to run.";
} catch (e) {
  status.textContent = `Could not load the WebAssembly module (${e}). Run scripts/build-playground.sh first.`;
}
