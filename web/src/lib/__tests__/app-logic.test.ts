import { describe, expect, it } from "vitest";

import { CLASSES, DISEASES, byCrop, diseaseForIndex } from "@/lib/classes";
import { headline, meanSd } from "@/lib/research";
import { CONFIDENT, MIN_PLANT_PIXELS, NOT_SURE, plantPixelShare, topK, verdict } from "@/lib/verdict";

function rgba(pixels: [number, number, number][]): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels.flatMap(([r, g, b]) => [r, g, b, 255]));
}

describe("class data", () => {
  it("has the 38 classes in the model's output order", () => {
    expect(CLASSES).toHaveLength(38);
    CLASSES.forEach((c, i) => expect(c.index).toBe(i));
    expect(CLASSES[0].class).toBe("Apple___Apple_scab");
    expect(CLASSES[37].class).toBe("Tomato___healthy");
  });

  it("has a library entry for every class, with unique slugs", () => {
    for (const c of CLASSES) expect(diseaseForIndex(c.index)?.class).toBe(c.class);
    expect(new Set(DISEASES.map((d) => d.slug)).size).toBe(DISEASES.length);
  });

  it("cites at least two sources for every disease", () => {
    for (const d of DISEASES.filter((x) => !x.healthy)) {
      expect(d.sources?.length ?? 0, d.class).toBeGreaterThanOrEqual(2);
      for (const s of d.sources ?? []) expect(s.url).toMatch(/^https:\/\//);
      expect(d.symptoms?.length ?? 0, d.class).toBeGreaterThan(0);
      expect(d.management?.length ?? 0, d.class).toBeGreaterThan(0);
    }
  });

  it("groups all classes by crop", () => {
    expect(byCrop().reduce((n, g) => n + g.diseases.length, 0)).toBe(38);
  });
});

describe("verdict", () => {
  it("ranks the top classes", () => {
    const top = topK([0.1, 0.6, 0.3], 2);
    expect(top.map((t) => t.index)).toEqual([1, 2]);
  });

  it("maps probability to confident / uncertain / not sure", () => {
    const leafy = MIN_PLANT_PIXELS + 0.1;
    expect(verdict([{ index: 0, probability: CONFIDENT }], leafy)).toBe("confident");
    expect(verdict([{ index: 0, probability: NOT_SURE }], leafy)).toBe("uncertain");
    expect(verdict([{ index: 0, probability: NOT_SURE - 0.01 }], leafy)).toBe("not-sure");
  });

  it("says 'not a leaf' when few pixels are plant coloured, even if confident", () => {
    expect(verdict([{ index: 0, probability: 0.99 }], MIN_PLANT_PIXELS / 2)).toBe("not-a-leaf");
  });

  it("counts green and brown pixels as plant, grey and blue as not", () => {
    expect(plantPixelShare(rgba([[40, 140, 50], [120, 80, 30]]))).toBe(1); // green, brown
    expect(plantPixelShare(rgba([[128, 128, 128], [30, 60, 200]]))).toBe(0); // grey, blue
    expect(plantPixelShare(rgba([[40, 140, 50], [128, 128, 128]]))).toBe(0.5);
  });
});

describe("research numbers", () => {
  it("parses mean ± sd strings", () => {
    expect(meanSd("98.26 ± 0.43")).toEqual({ mean: 98.26, sd: 0.43 });
    expect(meanSd("12.5")).toEqual({ mean: 12.5, sd: 0 });
  });

  it("finds the deployed model's headline numbers in research.json", () => {
    const h = headline();
    expect(h.labLeakFree.mean).toBeGreaterThan(90);
    expect(h.field.mean).toBeLessThan(h.labLeakFree.mean);
    expect(h.paperValLinked).toBeGreaterThan(0);
  });
});
