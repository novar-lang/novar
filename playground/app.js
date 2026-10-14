import init, { check, functions, run, hover } from "./pkg/telvax_wasm.js";
import { EXAMPLES } from "./examples.js";

const $ = (id) => document.getElementById(id);
const code = $("code"), highlight = $("highlight"), output = $("output"), problems = $("problems");
const status = $("status"), fnSelect = $("function"), argsBox = $("args"), exampleSelect = $("examples");

let ready = false;
let currentExample = EXAMPLES[0];

// ---- syntax highlighting (a light tokenizer; the compiler is the source of truth) ------------
const TOKEN = /(\/\/.*)|(f?"(?:[^"\\]|\\.)*")|\b(fn|struct|let|mut|pub|import|if|else|for|while|in|return|break|continue|match|enum|true|false)\b|\b(i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|bool|String|Vec|Vector|Tensor|Table|Result|[A-Z]\w*)\b|\b(\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)\b|(\|>|->|=>|\.\.=?|[-+*\/%=<>!&|]+)|(@\w+)|([A-Za-z_]\w*)(?=\s*\()/g;
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
