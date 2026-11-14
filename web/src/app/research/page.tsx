import type { Metadata } from "next";

import { AccuracyChart } from "@/components/accuracy-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { modelSizeMb } from "@/lib/model-info";
import { RESEARCH, type Row, meanSd, pct } from "@/lib/research";

export const metadata: Metadata = {
  title: "Research",
  description:
    "How LeafDx was built and evaluated: a leakage audit, a reproduction of the original paper, and lab-versus-field results.",
};

const REPO = "https://github.com/MoazamAli45/plant-leaf-disease-detection/blob/main";
const MODELS = ["Paper CNN (softmax)", "MobileNetV1", "MobileNetV2", "MobileNetV3-Small", "EfficientNetB0"];

export default function ResearchPage() {
  const a = RESEARCH.audit;
  const main = (RESEARCH.main_results as Row[]).filter((r) => MODELS.includes(r.model));
  const chart = main.map((r) => ({
    model: r.model,
    lab: meanSd(r["test acc., leak-free subset (%)"]),
    field: meanSd(r["PlantDoc acc. (%)"]),
  }));
  const repro = (RESEARCH.reproduction as Row[]).filter(
    (r) => r.schedule === "fullepochs" && /accuracy \(last epoch/.test(r.metric),
  );
  const field = RESEARCH.field_analysis as Row[];
  const explain = RESEARCH.explain as Record<string, Record<string, number>>;
  const eff = (RESEARCH.efficiency as Row[]).filter((r) => r.model === "p3_mobilenet_v1");
  const tfjs = RESEARCH.tfjs_verify as Record<string, { results: { agreement_with_keras: number; latency_ms_median: number; accuracy_test: number }[] }>;
  const tfjsSize = (RESEARCH.efficiency as Row[]).filter((r) => r.model === "p3_mobilenet_v1" && r.format.startsWith("TF.js"));
  const gaps = main.map((r) => meanSd(r["gap leak-free test - PlantDoc (pp)"]).mean);
  const gapRange = `${Math.floor(Math.min(...gaps))}–${Math.ceil(Math.max(...gaps))}`;
  const mnRepro = repro.find((r) => r.model === "mobilenet" && r.metric.startsWith("validation"));
  const cnnRepro = repro.find((r) => r.model === "cnn" && r.metric.startsWith("validation"));
  const uint8Changed = tfjs.tfjs_uint8 ? 1 - tfjs.tfjs_uint8.results[0].agreement_with_keras : null;
  const nExplain = Object.values(explain)[0]?.n ?? 0;
  const deployedMb = modelSizeMb();

  return (
    <div className="mx-auto max-w-4xl space-y-12">
      <header className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Research</h1>
        <p className="text-lg text-muted-foreground">
          LeafDx started from the paper <em>Plant Leaf Disease Detection Using Traditional CNN and
          MobileNet Architecture</em> (Ali Usman and Syed Moazam Ali, UET Taxila). This project
          audits its dataset, reproduces its two models, and tests how they and newer models
          behave on clean held-out data and on real field photos.
        </p>
        <p className="text-sm text-muted-foreground">
          Every number on this page is read from the project&apos;s result files (generated{" "}
          {RESEARCH.generated_at.slice(0, 10)}). Full write-ups:{" "}
          <a className="underline" href={`${REPO}/docs/dataset_card.md`}>dataset card</a>,{" "}
          <a className="underline" href={`${REPO}/docs/reproduction.md`}>reproduction</a>,{" "}
          <a className="underline" href={`${REPO}/docs/experiments.md`}>experiments</a>.
        </p>
      </header>

      <Section title="1. The dataset leaked">
        <p>
          The paper used the Kaggle <em>New Plant Diseases Dataset</em> ({a.npd_images.toLocaleString()}{" "}
          images, 38 classes), an offline-augmented copy of PlantVillage split 80/20 after
          augmentation. We traced every image to its source photo and physical leaf.
        </p>
        <Stats
          items={[
            [pct(a.paper_val_linked_frac), `of ${a.paper_val_images.toLocaleString()} validation images have a rotated/flipped copy or another photo of the same leaf in training`],
            [pct(a.paper_val_source_frac), "share an augmented copy of the same photo"],
            [pct(a.paper_val_leaf_frac), "show a leaf that also appears in training"],
          ]}
        />
        <p className="text-sm text-muted-foreground">
          We built a grouped 70/15/15 split ({Object.entries(a.clean_split_counts).map(([k, v]) => `${k} ${Number(v).toLocaleString()}`).join(", ")}).
          A later check found one augmentation type our parser missed, which left {pct(a.clean_test_linked_frac_v1)} of
          its test images linked to training; results below use the leak-free remainder.
        </p>
      </Section>

      <Section title="2. Reproducing the paper">
        <p>Both models rebuilt from the paper&apos;s description and trained on its original split (one run each, as in the paper):</p>
        <SimpleTable
          head={["Model", "Metric", "Paper", "Reproduced", "Difference (pp)"]}
          rows={repro.map((r) => [r.model === "cnn" ? "Custom CNN" : "MobileNet", r.metric.replace(" (last epoch, as Keras reports)", ""), num(r.paper), num(r.reproduced), num(r.difference)])}
        />
        <p className="text-sm text-muted-foreground">
          Validation accuracy: MobileNet differs from the paper by {num(mnRepro?.difference ?? "")} points,
          the CNN by {num(cnnRepro?.difference ?? "")}, plausibly because the paper does not state
          augmentation strengths or optimiser settings. These validation scores are inflated by the
          leakage above.
        </p>
      </Section>

      <Section title="3. Lab accuracy versus field accuracy">
        <p>
          Retrained on the clean split (3 seeds each) and tested on held-out lab photos and on{" "}
          {a.plantdoc_test_images.toLocaleString()} field photos from PlantDoc mapped to the same classes:
        </p>
        <AccuracyChart data={chart} />
        <SimpleTable
          head={["Model", "Params (M)", "Lab test, leak-free (%)", "Field (%)", "Field calibration error (%)"]}
          rows={main.map((r) => [r.model, r["params (M)"], r["test acc., leak-free subset (%)"], r["PlantDoc acc. (%)"], r["PlantDoc ECE (%)"]])}
        />
        <p>
          Every model is {gapRange} points worse on field photos. Higher lab accuracy does not close the
          gap, and the models stay confident when they are wrong.
        </p>
      </Section>

      <Section title="4. How the models fail in the field">
        <SimpleTable
          head={["Model", "Correct class", "Correct crop", "Healthy vs diseased", "Confidence when right / wrong"]}
          rows={field.map((r) => [r.model, r.accuracy, r.crop_accuracy, r.healthy_vs_diseased_accuracy, `${r.mean_conf_correct.split(" ")[0]} / ${r.mean_conf_wrong.split(" ")[0]}`])}
        />
        <p className="text-sm text-muted-foreground">
          Percentages, mean ± std over seeds where shown. Always answering &quot;diseased&quot; would
          score {field[0]?.always_diseased_baseline.split(" ")[0]}% on healthy vs diseased.
        </p>
      </Section>

      <Section title="5. Where the models look">
        <SimpleTable
          head={["Model", "Grad-CAM on leaf ÷ leaf area", "Leaf only", "Background only", "Border strip only"]}
          rows={Object.entries(explain).map(([k, v]) => [
            labelFor(k),
            v.leaf_energy_ratio_median.toFixed(2),
            pct(v.leaf_only_accuracy),
            pct(v.background_only_accuracy),
            pct(v.border_only_accuracy),
          ])}
        />
        <p className="text-sm text-muted-foreground">
          {nExplain} test photos with the PlantVillage authors&apos; leaf masks; chance is 2.6%. Fine-tuned
          models focus on the leaf and its lesions; a weaker background signal remains. The
          paper&apos;s CNN barely prefers the leaf.
        </p>
      </Section>

      <Section title="6. The model in this app">
        <p>
          MobileNetV1 was chosen for the browser: it generalises best to field photos among the
          small models and survives compression well.
        </p>
        <SimpleTable
          head={["Format", "Size (MB)", "CPU latency (ms)", "Accuracy (%)"]}
          rows={eff.filter((r) => r.cpu_latency_ms_median).map((r) => [r.format, Number(r.size_mb).toFixed(2), Number(r.cpu_latency_ms_median).toFixed(2), (Number(r.accuracy) * 100).toFixed(2)])}
        />
        <SimpleTable
          head={["TF.js weights", "Size (MB)", "Same answer as Keras", "WASM latency in Node (ms)"]}
          rows={["tfjs", "tfjs_fp16", "tfjs_uint8"].filter((k) => tfjs[k]).map((k) => {
            const r = tfjs[k].results[0];
            const size = k === "tfjs_fp16" ? `${deployedMb} (deployed)` : sizeFor(tfjsSize, k);
            return [k.replace("tfjs_", "").replace("tfjs", "float32"), size, pct(r.agreement_with_keras), r.latency_ms_median.toFixed(1)];
          })}
        />
        <p className="text-sm text-muted-foreground">
          uint8 weights would be smallest but change{" "}
          {uint8Changed === null ? "some" : pct(uint8Changed, 0)} of answers, so the app ships float16.
          CPU latency measured on an Apple M3 Pro; browser speed depends on your device.
        </p>
      </Section>

      <Section title="7. Ablations">
        {RESEARCH.ablations ? (
          <SimpleTable
            head={["Variant", "Lab test, leak-free (%)", "Field (%)"]}
            rows={(RESEARCH.ablations as Row[]).map((r) => [r.model, r["test acc., leak-free subset (%)"], r["PlantDoc acc. (%)"]])}
          />
        ) : (
          <p className="text-muted-foreground">
            Pending: augmentation off, frozen backbone (the paper&apos;s set-up) and smaller width
            multipliers for MobileNetV1 are still training.
          </p>
        )}
      </Section>
    </div>
  );
}

function labelFor(key: string): string {
  return (
    {
      p3_cnn_softmax: "Paper CNN (softmax)",
      p3_mobilenet_v1: "MobileNetV1",
      p3_mobilenet_v2: "MobileNetV2",
      p3_mobilenet_v3_small: "MobileNetV3-Small",
      p3_efficientnet_b0: "EfficientNetB0",
    } as Record<string, string>
  )[key] ?? key;
}

function sizeFor(rows: Row[], key: string): string {
  const fmt = key === "tfjs_uint8" ? "TF.js uint8 weights" : "TF.js float32";
  const r = rows.find((x) => x.format === fmt);
  return r ? Number(r.size_mb).toFixed(1) : "–";
}

function num(v: string): string {
  if (!v) return "–";
  const n = Number(v);
  return Number.isFinite(n) ? (Math.abs(n) < 1 ? (n * 100).toFixed(2) : n.toFixed(2)) : v;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function Stats({ items }: { items: [string, string][] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {items.map(([v, l]) => (
        <Card key={l}>
          <CardHeader className="pb-1">
            <CardTitle className="text-2xl tabular-nums">{v}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">{l}</CardContent>
        </Card>
      ))}
    </div>
  );
}

function SimpleTable({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            {head.map((h) => (
              <TableHead key={h}>{h}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i}>
              {r.map((c, j) => (
                <TableCell key={j} className={j === 0 ? "font-medium" : "tabular-nums"}>
                  {c}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
