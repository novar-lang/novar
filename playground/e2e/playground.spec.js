// @ts-check
const { test, expect } = require("@playwright/test");

test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/index.html");
  // The compiler is a Wasm module: wait until the page says it is ready.
  await expect(page.locator("#status")).not.toHaveText(/Loading the compiler/, { timeout: 15_000 });
  // @ts-ignore
  page.errors = errors;
});

test.afterEach(async ({ page }) => {
  // @ts-ignore
  expect(page.errors, "the page must not log errors").toEqual([]);
});

async function setCode(page, source) {
  await page.locator("#code").fill(source);
  // Checking is debounced; wait for the problems list to be rebuilt.
  await page.waitForTimeout(400);
}

test("loads with examples and the first one checks cleanly", async ({ page }) => {
  expect(await page.locator("#examples option").count()).toBeGreaterThan(2);
  await expect(page.locator("#problems")).toContainText("No problems");
  await expect(page.locator("#code")).toHaveValue(/fn main/);
});

test("runs the default example", async ({ page }) => {
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("hello, world");
  await expect(page.locator("#output")).toContainText("=> 0");
});

test("Ctrl+Enter runs the program", async ({ page }) => {
  await setCode(page, 'fn main() -> i32:\n    print("from the keyboard")\n    7\n');
  await page.locator("#code").press("Control+Enter");
  await expect(page.locator("#output")).toContainText("from the keyboard");
  await expect(page.locator("#output")).toContainText("=> 7");
});

test("shows problems for a program with a type error", async ({ page }) => {
  await setCode(page, 'fn main() -> i32:\n    "not a number"\n');
  await expect(page.locator("#problems li.error").first()).toContainText("mismatched types");
  await expect(page.locator("#problem-count")).toContainText("(");
});

test("a runtime error is reported in the output", async ({ page }) => {
  await setCode(page, "fn main() -> i32:\n    let d = 0\n    10 / d\n");
  await page.locator("#run").click();
  await expect(page.locator("#output .error")).toContainText("error:");
});

test("runs enums, match, Option and Map", async ({ page }) => {
  await setCode(
    page,
    `enum Shape:
    Circle(Float)
    Empty

fn area(s: Shape) -> Float:
    match s:
        case Shape.Circle(r):
            r * r * 3.0
        case Shape.Empty:
            0.0

fn main() -> i32:
    let m: Map[Text, Int] = std.map.new()
    m.insert("a", 2)
    var n = 0
    if m.get("a") is Some(v):
        n = v
    print(f"area {area(Shape.Circle(2.0))} and {n}")
    0
`,
  );
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("area 12.0 and 2");
});

test("picking another example loads and runs it", async ({ page }) => {
  await page.locator("#examples").selectOption({ index: 1 });
  await expect(page.locator("#problems")).toContainText("No problems");
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("finished in");
});

test("the share link carries the program", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await setCode(page, "fn main() -> i32:\n    99\n");
  await page.locator("#share").click();
  await expect(page.locator("#status")).toContainText(/Link copied|share/i);
  const url = page.url();
  expect(url).toContain("#");
  await page.goto(url);
  await page.reload();
  await expect(page.locator("#code")).toHaveValue("fn main() -> i32:\n    99\n");
});

// ---- the console ----------------------------------------------------------------------------

/** Types a line into the console and presses Enter. */
async function type(page, line) {
  const input = page.locator("#console-input");
  await input.fill(line);
  await input.press("Enter");
}

test("the console keeps variables and definitions between lines", async ({ page }) => {
  await type(page, "let x = 5");
  await type(page, "x * 2");
  await expect(page.locator("#console-log")).toContainText("10");
  await type(page, "fn double(k: Int) -> Int:");
  await expect(page.locator("#console-prompt")).toHaveText("...");
  // The next line starts indented; an empty line ends the block.
  await expect(page.locator("#console-input")).toHaveValue("    ");
  await type(page, "    k * 2");
  await type(page, "");
  await expect(page.locator("#console-prompt")).toHaveText(">>>");
  await type(page, "double(x)");
  await expect(page.locator("#console-log .ok").last()).toHaveText("10");
});

test("the console shows what is printed and the line of an error", async ({ page }) => {
  await type(page, 'print("from the console")');
  await expect(page.locator("#console-log")).toContainText("from the console");
  await type(page, 'let y = 1');
  await type(page, 'y + "a"');
  await expect(page.locator("#console-log .error").last()).toContainText("mismatched operand types");
  await expect(page.locator("#console-log .error").last()).toContainText('1 | y + "a"');
  // The session is as it was.
  await type(page, "y");
  await expect(page.locator("#console-log .ok").last()).toHaveText("1");
});

test("the console has commands, history and Tab", async ({ page }) => {
  await type(page, "let origin = [1, 2, 3]");
  await type(page, ":type origin");
  await expect(page.locator("#console-log .ok").last()).toHaveText("List[Int]");
  await type(page, ":vars");
  await expect(page.locator("#console-log")).toContainText("origin: List[Int] = [1, 2, 3]");
  // The arrow keys bring back earlier lines.
  const input = page.locator("#console-input");
  await input.press("ArrowUp");
  await expect(input).toHaveValue(":vars");
  await input.press("ArrowUp");
  await expect(input).toHaveValue(":type origin");
  await input.fill("ori");
  await input.press("Tab");
  await expect(input).toHaveValue("origin");
  await input.fill("");
  await type(page, ":reset");
  await type(page, "origin");
  await expect(page.locator("#console-log .error").last()).toBeVisible();
});

test("the program in the editor can be loaded into the console", async ({ page }) => {
  await setCode(page, "fn triple(n: Int) -> Int:\n    n * 3\n\nfn main() -> Int:\n    0\n");
  await page.locator("#console-load").click();
  await expect(page.locator("#console-log")).toContainText("entered");
  await type(page, "triple(14)");
  await expect(page.locator("#console-log .ok").last()).toHaveText("42");
});

test("a runaway input in the console stops and the console goes on", async ({ page }) => {
  await type(page, "while True:");
  await type(page, "    pass");
  await type(page, "");
  await expect(page.locator("#console-log .error").last()).toContainText("error", { timeout: 20_000 });
  await type(page, "1 + 1");
  await expect(page.locator("#console-log .ok").last()).toHaveText("2");
});
