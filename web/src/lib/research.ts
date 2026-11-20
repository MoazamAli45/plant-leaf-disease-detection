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

/** A Phase 6 model fine-tuned on field photos, when one is deployed (docs/field.md). */
export type FieldAdapted = {
  base_model: string;
  k_per_class: number | string;
  seed: number;
  n_field_train: number;
  n_lab_replay: number;
  tta?: boolean;
  lab_test_leak_free: { n: number; accuracy: number };
  field_test: { n: number; accuracy: number; crop_accuracy: number; ece: number };
  tfjs_fp16_mb: number;
};

export const FIELD_ADAPTED = ((RESEARCH as { deployed?: FieldAdapted | null }).deployed ?? null) as FieldAdapted | null;

export function headline() {
  const mn = mainResult(DEPLOYED);
  if (!mn) throw new Error(`${DEPLOYED} missing from research.json`);
  const common = { params: mn["params (M)"], paperValLinked: RESEARCH.audit.paper_val_linked_frac };
  if (FIELD_ADAPTED) {
    return {
      ...common,
      labLeakFree: { mean: FIELD_ADAPTED.lab_test_leak_free.accuracy * 100, sd: 0 },
      field: { mean: FIELD_ADAPTED.field_test.accuracy * 100, sd: 0 },
      fieldImages: FIELD_ADAPTED.field_test.n,
      fieldSet: "held-out field photos (PlantDoc test split)",
      modelLabel: `Deployed model: MobileNetV1 fine-tuned on labelled field photos${FIELD_ADAPTED.tta ? ", averaging 4 views per photo" : ""} (one training run).`,
      fieldAdapted: true,
    };
  }
  return {
    ...common,
    labLeakFree: meanSd(mn["test acc., leak-free subset (%)"]),
    field: meanSd(mn["PlantDoc acc. (%)"]),
    fieldImages: RESEARCH.audit.plantdoc_test_images,
    fieldSet: "real field photos (PlantDoc)",
    modelLabel: "Deployed model: MobileNetV1, mean of 3 training runs.",
    fieldAdapted: false,
  };
}
