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
  await expect(page.locator("#code")).toHaveValue(/def main/);
});

test("runs the default example", async ({ page }) => {
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("hello, world");
  await expect(page.locator("#output")).toContainText("=> 0");
});

test("Ctrl+Enter runs the program", async ({ page }) => {
  await setCode(page, 'def main() -> int32:\n    print("from the keyboard")\n    return 7\n');
  await page.locator("#code").press("Control+Enter");
  await expect(page.locator("#output")).toContainText("from the keyboard");
  await expect(page.locator("#output")).toContainText("=> 7");
});

test("shows problems for a program with a type error", async ({ page }) => {
  await setCode(page, 'def main() -> int32:\n    return "not a number"\n');
  await expect(page.locator("#problems li.error").first()).toContainText("mismatched types");
  await expect(page.locator("#problem-count")).toContainText("(");
});

test("a runtime error is reported in the output", async ({ page }) => {
  await setCode(page, "def main() -> int32:\n    d = 0\n    return 10 / d\n");
  await page.locator("#run").click();
  await expect(page.locator("#output .error")).toContainText("error:");
});

test("runs enums, match, Option and Map", async ({ page }) => {
  await setCode(
    page,
    `enum Shape:
    Circle(float)
    Empty

def area(s: Shape) -> float:
    match s:
        case Shape.Circle(r):
            return r * r * 3.0
        case Shape.Empty:
            return 0.0

def main() -> int32:
    m: dict[str, int] = std.map.new()
    m.insert("a", 2)
    n = 0
    if m.get("a") is Some(v):
        n = v
    print(f"area {area(Shape.Circle(2.0))} and {n}")
    return 0
`,
  );
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("area 12 and 2");
});

test("picking another example loads and runs it", async ({ page }) => {
  await page.locator("#examples").selectOption({ index: 1 });
  await expect(page.locator("#problems")).toContainText("No problems");
  await page.locator("#run").click();
  await expect(page.locator("#output")).toContainText("finished in");
});

test("the share link carries the program", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await setCode(page, "def main() -> int32:\n    return 99\n");
  await page.locator("#share").click();
  await expect(page.locator("#status")).toContainText(/Link copied|share/i);
  const url = page.url();
  expect(url).toContain("#");
  await page.goto(url);
  await page.reload();
  await expect(page.locator("#code")).toHaveValue("def main() -> int32:\n    return 99\n");
});
