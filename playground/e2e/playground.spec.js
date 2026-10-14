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
  await setCode(page, 'fn main() -> i32 { std::io::println("from the keyboard"); 7 }');
  await page.locator("#code").press("Control+Enter");
  await expect(page.locator("#output")).toContainText("from the keyboard");
  await expect(page.locator("#output")).toContainText("=> 7");
});

test("shows problems for a program with a type error", async ({ page }) => {
  await setCode(page, 'fn main() -> i32 { "not a number" }');
  await expect(page.locator("#problems li.error").first()).toContainText("mismatched types");
  await expect(page.locator("#problem-count")).toContainText("(");
});

test("a runtime error is reported in the output", async ({ page }) => {
  await setCode(page, "fn main() -> i32 { let d = 0; 10 / d }");
  await page.locator("#run").click();
  await expect(page.locator("#output .error")).toContainText("error:");
});

test("runs enums, match, Option and Map", async ({ page }) => {
  await setCode(
    page,
    `enum Shape { Circle(f64), Empty }
fn area(s: Shape) -> f64 { match s { Shape::Circle(r) => r * r * 3.0, Shape::Empty => 0.0 } }
fn main() -> i32 {
    let m: Map<String, i64> = std::map::new();
    m.insert("a", 2);
    let n = match m.get("a") { Some(v) => v, None => 0 };
    std::io::println(f"area {area(Shape::Circle(2.0))} and {n}");
    0
}`,
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
  await setCode(page, "fn main() -> i32 { 99 }");
  await page.locator("#share").click();
  await expect(page.locator("#status")).toContainText(/Link copied|share/i);
  const url = page.url();
  expect(url).toContain("#");
  await page.goto(url);
  await page.reload();
  await expect(page.locator("#code")).toHaveValue("fn main() -> i32 { 99 }");
});
