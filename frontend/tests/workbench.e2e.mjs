import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { chromium } from "playwright-core";

// The host serves the production frontend and an isolated local SQLite task store.
// The test enqueues offline runs. No model keys or paid requests are required.
const origin = new URL(process.env.DATAVERIFY_WORKBENCH_URL || "http://127.0.0.1:8765");
if (!["localhost", "127.0.0.1", "[::1]"].includes(origin.hostname)) {
  throw new Error("Browser acceptance runs require a loopback service.");
}
const counts = "统计所有已结算明细条数，相同金额的不同明细分别计数。";
const total = "统计所有已结算明细的净金额，退款按负值计入。";
const ambiguous = "统计已结算的数量。";

function taskDetail(page, id) {
  return page.locator("article.result").filter({ has: page.getByText(id, { exact: true }) });
}

test("a visitor can inspect the public source and download checked full-dataset results", { timeout: 90_000 }, async t => {
  const browser = await chromium.launch(process.env.CHROME_EXECUTABLE
    ? { executablePath: process.env.CHROME_EXECUTABLE } : { channel: "chrome" });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const sourceUrl = new URL("/?source=palmer-penguins", origin);
  await page.goto(sourceUrl.href);
  const source = page.getByLabel("数据源", { exact: true });
  await source.locator('option[value="palmer-penguins"]').waitFor({ state: "attached" });
  assert.equal(await source.inputValue(), "palmer-penguins");
  assert.match(await page.locator(".composer .meta a").getAttribute("href"), /palmerpenguins/);
  await page.getByText("查看原始样本、表结构与统计口径").click();
  const preview = page.getByRole("table", { name: "penguins 原始样本" });
  await preview.locator("tbody td").first().waitFor();
  assert.equal(await preview.locator("tbody tr").count(), 5);
  assert.ok(await preview.getByRole("columnheader", { name: "body_mass_g" }).isVisible());
  assert.ok(await preview.getByText("Adelie", { exact: true }).count() > 0);

  await page.getByRole("button", { name: "分别统计每个物种的观测记录数，按物种名称升序排列。" }).click();
  await page.getByRole("button", { name: "提交核验", exact: true }).click();
  await page.waitForURL(/#\/run\//);
  const rows = page.locator("article.result table tbody tr");
  await rows.first().waitFor({ timeout: 30_000 });
  assert.deepEqual(await rows.allTextContents(), ["Adelie152", "Chinstrap68", "Gentoo124"]);
  const downloaded = page.waitForEvent("download");
  await page.getByRole("link", { name: "下载 CSV" }).click();
  const download = await downloaded;
  assert.match(download.suggestedFilename(), /^dataverify-.+\.csv$/);
  const csv = await readFile(await download.path(), "utf8");
  assert.deepEqual(csv.trim().split(/\r?\n/).slice(1), ["Adelie,152", "Chinstrap,68", "Gentoo,124"]);
  await page.getByRole("link", { name: "← 新建任务" }).click();
  await source.selectOption("demo-orders");
  assert.equal(new URL(page.url()).searchParams.get("source"), "demo-orders");
  assert.equal(await page.getByLabel("统计问题").inputValue(), "");
  assert.equal(await page.getByRole("table", { name: "penguins 原始样本" }).count(), 0);
  await page.getByRole("link", { name: "2009 年有多少条企鹅观测" }).click();
  await page.waitForURL(/#\/replay\/penguins-2009$/);
  const recorded = taskDetail(page, "penguins-2009");
  await recorded.locator("td").first().waitFor();
  assert.equal(await recorded.locator("td").first().innerText(), "120");
  assert.ok(await recorded.getByText("原任务用量 · 5 次千问请求 · 14,105 tokens").isVisible());
  assert.match(await recorded.locator(".sql").innerText(), /year = 2009/i);
});

test("workbench submits and inspects local SQL runs, resumes a clarified question, and opens recorded evidence", {timeout: 120_000}, async t => {
  const browser = await chromium.launch(process.env.CHROME_EXECUTABLE
    ? { executablePath: process.env.CHROME_EXECUTABLE }
    : { channel: "chrome" });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const submitExample = async question => {
    await page.goto(origin.href);
    await page.getByLabel("执行模式").selectOption("offline");
    await page.getByRole("button", { name: question, exact: true }).click();
    await page.getByRole("button", { name: "提交核验", exact: true }).click();
    await page.waitForURL(/#\/run\//);
    return page.url();
  };
  const expectResult = async (value, id = new URL(page.url()).hash.split("/").pop()) => {
    const result = taskDetail(page, id);
    await result.locator("td").first().waitFor({timeout: 25_000});
    assert.equal(await result.locator(".status").innerText(), "有限核验通过");
    assert.equal(await result.locator("td").first().innerText(), String(value));
  };

  await page.goto(origin.href);
  const mode = page.getByLabel("执行模式");
  const liveOption = mode.locator('option[value="qwen"]');
  await liveOption.waitFor({state: "attached"});
  if (await liveOption.isEnabled()) {
    await mode.selectOption("qwen");
    const investigate = page.getByRole("checkbox", {name: "分歧时允许额外查询"});
    assert.equal(await investigate.isChecked(), false);
    const split = page.getByRole("checkbox", {name: "试用分栏签核"});
    assert.equal(await split.isChecked(), false);
    await split.check();
    await investigate.check();
    assert.equal(await investigate.isChecked(), true);
  }

  await submitExample(total);
  await expectResult(265);
  assert.ok(await page.getByText("离线演示 · 0 次模型 API 请求", {exact: true}).isVisible());
  const runUrl = page.url();
  await page.reload();
  await expectResult(265);
  assert.equal(page.url(), runUrl);
  assert.match(await page.locator(".sql").innerText(), /SUM/i);

  await submitExample(counts);
  await expectResult(4);
  const citations = page.getByRole("region", {name: "交付引用"});
  assert.ok(await citations.getByRole("heading", {name:"签核依据"}).isVisible());
  assert.match(await citations.locator("code").first().innerText(), /^replay_/);
  assert.equal(await citations.locator("code").last().innerText(), "q0001");
  await page.getByText("执行只读调查查询", {exact: true}).first().click();
  assert.match(await page.locator("details.evidence[open]").first().innerText(), /COUNT|SELECT/i);
  const revision = page.locator("details.evidence summary").filter({hasText: "候选 SQL · 已采用"});
  await revision.first().click();
  assert.match(await revision.first().locator("..").innerText(), /COUNT\(DISTINCT/i);

  const parentUrl = await submitExample(ambiguous);
  await page.getByLabel("补充业务定义").fill("按订单数");
  await page.getByRole("button", {name:"提交并继续"}).click();
  await page.waitForURL(url => url.href !== parentUrl && url.hash.startsWith("#/run/"));
  await expectResult(2);
  const childUrl = page.url();
  await page.getByRole("link", {name:"查看上一次任务"}).click();
  await page.getByRole("link", {name:"查看补充口径后的任务"}).click();
  assert.equal(page.url(), childUrl);

  await page.getByRole("navigation", {name:"历史回放"}).getByRole("link", {name:"等额明细被重复去重"}).click();
  await expectResult(6, "duplicate-amount");
  assert.ok(await page.getByText("已保存的历史回放", {exact: true}).isVisible());
  assert.ok(await page.getByText("历史模型记录 · 本次查看新增 0 次 API 请求", {exact: true}).isVisible());
  await page.getByText("初始分歧", {exact: true}).click();
  assert.match(await page.locator("details.evidence[open]").first().innerText(), /5.*5.*6/);
  await page.getByRole("navigation", {name:"历史回放"}).locator('a[href="#/replay/split-signoff"]').click();
  await expectResult(669, "split-signoff");
  const realCitations = page.getByRole("region", {name: "交付引用"});
  assert.match(await realCitations.locator("code").first().innerText(), /^replay_.*:data_/);
  assert.equal(await realCitations.locator("code").last().innerText(), "q0002");
  assert.deepEqual(errors, []);
});
