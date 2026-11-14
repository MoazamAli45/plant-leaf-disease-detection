/**
 * Accessors for research.json, which is generated from the ML result files by
 * `python -m leafdx.train.export_web research`. Pages must take every number from here.
 */
import research from "@/data/research.json";

export const RESEARCH = research;

export type Row = Record<string, string>;

/** "98.26 ± 0.43" -> { mean: 98.26, sd: 0.43 } */
export function meanSd(value: string): { mean: number; sd: number } {
  const [mean, sd] = value.split("±").map((s) => Number.parseFloat(s.trim()));
  return { mean, sd: Number.isFinite(sd) ? sd : 0 };
}

export function mainResult(model: string): Row | undefined {
  return (RESEARCH.main_results as Row[]).find((r) => r.model === model);
}

export function pct(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** The deployed model and its headline numbers. */
export const DEPLOYED = "MobileNetV1";

export function headline() {
  const mn = mainResult(DEPLOYED);
  if (!mn) throw new Error(`${DEPLOYED} missing from research.json`);
  return {
    labLeakFree: meanSd(mn["test acc., leak-free subset (%)"]),
    field: meanSd(mn["PlantDoc acc. (%)"]),
    params: mn["params (M)"],
    paperValLinked: RESEARCH.audit.paper_val_linked_frac,
    fieldImages: RESEARCH.audit.plantdoc_test_images,
  };
}
