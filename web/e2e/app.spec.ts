import fs from "node:fs";
import path from "node:path";

import { expect, type Page, test } from "@playwright/test";

const FIXTURES = path.join(__dirname, "fixtures");

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  return errors;
}

async function analyse(page: Page, file: string) {
  await page.goto("/detect");
  // wait for the model to finish loading (progress text disappears)
  await expect(page.getByText(/Loading the model/)).toBeHidden({ timeout: 90_000 });
  await page.locator('input[type="file"]').setInputFiles(path.join(FIXTURES, file));
  await page.getByRole("button", { name: "Analyse this leaf" }).click();
  await page.waitForURL("**/result");
}

for (const route of ["/", "/detect", "/library", "/library/tomato-early-blight", "/research", "/about"]) {
  test(`renders ${route} without errors`, async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto(route);
    await expect(page.locator("main h1").first()).toBeVisible();
    // no horizontal scrolling at this viewport
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    expect(errors).toEqual([]);
  });
}

test("library lists all 38 classes", async ({ page }) => {
  await page.goto("/library");
  await expect(page.locator('main a[href^="/library/"]')).toHaveCount(38);
});

test("research page shows the lab-vs-field chart", async ({ page }) => {
  await page.goto("/research");
  await expect(page.getByRole("img", { name: /Accuracy by model/ })).toBeVisible();
});

const leaves = JSON.parse(fs.readFileSync(path.join(FIXTURES, "fixtures.json"), "utf8")) as Record<
  string,
  { class: string | null }
>;
const expectedName: Record<string, RegExp> = {
  Tomato___Early_blight: /Early blight of tomato/,
  Apple___Apple_scab: /Apple scab/,
  Grape___healthy: /Healthy grape leaf/,
};

for (const [file, meta] of Object.entries(leaves)) {
  const cls = meta?.class;
  if (!cls) continue;
  test(`classifies ${file} in the browser`, async ({ page }, info) => {
    const errors = collectErrors(page);
    await analyse(page, file);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(expectedName[cls]);
    await expect(page.getByText(/confidence/).first()).toBeVisible();
    const details = await page.getByText(/Analysed in your browser/).textContent();
    info.annotations.push({ type: "inference", description: details ?? "" });
    expect(errors).toEqual([]);
  });
}

test("a grey photo is reported as not a leaf", async ({ page }) => {
  await analyse(page, "grey-wall.jpg");
  await expect(page.getByText("This doesn't look like a leaf")).toBeVisible();
});

test("result page without an analysis asks for a photo", async ({ page }) => {
  await page.goto("/result");
  await expect(page.getByText("No result yet")).toBeVisible();
});

test("heat map is computed on the device, without network requests", async ({ page }) => {
  const errors = collectErrors(page);
  await analyse(page, "apple-apple-scab.jpg");
  const requests: string[] = [];
  page.on("request", (r) => requests.push(r.url()));
  await page.getByRole("button", { name: "Show heat map" }).click();
  const overlay = page.getByRole("img", { name: /Heat map over the photo/ });
  await expect(overlay).toBeVisible({ timeout: 30_000 });
  expect(await overlay.getAttribute("src")).toMatch(/^data:image\/jpeg;base64,/);
  // only same-origin assets (e.g. cached model files) may be fetched; the photo is never sent anywhere
  expect(requests.filter((u) => !u.startsWith("data:") && !u.startsWith("blob:") && !u.includes("localhost"))).toEqual([]);
  expect(requests.filter((u) => /predict|gradcam/i.test(u))).toEqual([]);
  expect(errors).toEqual([]);
});
