"use client";

import { AlertTriangle, CheckCircle2, Eye, HelpCircle, ImageOff, Loader2 } from "lucide-react";
import Link from "next/link";
import { useMemo, useState, useSyncExternalStore } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CLASSES, DISEASE_META, type Disease, diseaseForIndex } from "@/lib/classes";
import { colourise } from "@/lib/gradcam";
import { INPUT_SIZE, explain } from "@/lib/model";
import { parseAnalysis, readAnalysisRaw } from "@/lib/storage";
import type { Verdict } from "@/lib/verdict";

const subscribe = () => () => {};

export function ResultView({ fieldAccuracy }: { fieldAccuracy: string }) {
  // client-only read; the server render shows the loading state
  const raw = useSyncExternalStore(subscribe, readAnalysisRaw, () => undefined);
  const analysis = useMemo(() => (raw === undefined ? undefined : parseAnalysis(raw)), [raw]);

  if (analysis === undefined) {
    return <p className="text-muted-foreground">Loading result…</p>;
  }
  if (analysis === null) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No result yet</CardTitle>
          <CardDescription>Results are kept only in this browser tab.</CardDescription>
        </CardHeader>
        <CardContent>
          <Button asChild>
            <Link href="/detect">Check a leaf</Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const top = analysis.top[0];
  const disease = diseaseForIndex(top.index);
  const showInfo = analysis.verdict === "confident" || analysis.verdict === "uncertain";

  return (
    <div className="space-y-6">
      <VerdictBanner verdict={analysis.verdict} disease={disease} probability={top.probability} />
      <div className="grid gap-8 lg:grid-cols-[minmax(0,20rem)_1fr]">
      <div className="space-y-4">
        {/* eslint-disable-next-line @next/next/no-img-element -- local data URL */}
        <img src={analysis.image} alt="The analysed photo" className="w-full rounded-lg border object-contain" />
        <p className="text-xs text-muted-foreground">
          Analysed in your browser ({analysis.backend}, {Math.round(analysis.ms)} ms).
        </p>
        <Button asChild variant="outline" className="w-full">
          <Link href="/detect">Check another leaf</Link>
        </Button>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Most likely classes</CardTitle>
            <CardDescription>The model&apos;s probabilities for its top three answers.</CardDescription>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3">
              {analysis.top.map((p) => {
                const d = diseaseForIndex(p.index);
                const pct = Math.round(p.probability * 1000) / 10;
                const label = formatProbability(p.probability);
                return (
                  <li key={p.index} className="space-y-1">
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span>
                        {d ? (
                          <Link href={`/library/${d.slug}`} className="underline-offset-2 hover:underline">
                            {d.name}
                          </Link>
                        ) : (
                          CLASSES[p.index]?.name
                        )}
                        {d && <span className="text-muted-foreground"> · {d.crop}</span>}
                      </span>
                      <span className="tabular-nums text-muted-foreground">{label}</span>
                    </div>
                    <div className="h-2 rounded-full bg-muted" aria-hidden>
                      <div className="h-2 rounded-full bg-primary" style={{ width: `${Math.max(pct, 1)}%` }} />
                    </div>
                  </li>
                );
              })}
            </ol>
          </CardContent>
        </Card>

        <Alert>
          <AlertTriangle aria-hidden />
          <AlertTitle>Read this before acting on the result</AlertTitle>
          <AlertDescription>
            On real field photos this model was right only {fieldAccuracy} of the time in our
            evaluation. It is a learning tool, not a diagnosis. {DISEASE_META.disclaimer}
          </AlertDescription>
        </Alert>

        {showInfo && disease && <DiseaseInfo disease={disease} />}

        <GradcamPanel image={analysis.image} classIndex={top.index} label={disease?.name ?? CLASSES[top.index]?.name ?? "this class"} />
      </div>
      </div>
    </div>
  );
}

/** Probabilities as the user sees them; never "100%" or "0%" (the model is never certain). */
export function formatProbability(p: number): string {
  if (p >= 0.995) return "over 99%";
  if (p < 0.005) return "under 1%";
  return `${Math.round(p * 100)}%`;
}

function VerdictBanner({ verdict, disease, probability }: { verdict: Verdict; disease?: Disease; probability: number }) {
  const pct = formatProbability(probability);
  if (verdict === "not-a-leaf") {
    return (
      <Alert>
        <ImageOff aria-hidden />
        <AlertTitle>This doesn&apos;t look like a leaf</AlertTitle>
        <AlertDescription>
          Very little of the photo has plant colours. Try a close-up of a single leaf on a plain
          background.
        </AlertDescription>
      </Alert>
    );
  }
  if (verdict === "not-sure") {
    return (
      <Alert>
        <HelpCircle aria-hidden />
        <AlertTitle>Not sure</AlertTitle>
        <AlertDescription>
          The model&apos;s best guess has only {pct} probability, too low to show. The leaf may be
          from a crop or condition the model does not know, or the photo may be unclear.
        </AlertDescription>
      </Alert>
    );
  }
  const name = !disease
    ? "Unknown"
    : disease.name.toLowerCase().includes(disease.crop.split(" ")[0].toLowerCase())
      ? disease.name
      : `${disease.name} (${disease.crop})`;
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2">
        {verdict === "confident" ? (
          <CheckCircle2 className="size-5 text-primary" aria-hidden />
        ) : (
          <HelpCircle className="size-5 text-orange-700 dark:text-orange-400" aria-hidden />
        )}
        <h1 className="text-2xl font-semibold tracking-tight">{name}</h1>
        <Badge variant={verdict === "confident" ? "default" : "secondary"}>
          {verdict === "confident" ? `${pct} confidence` : `low confidence: ${pct}`}
        </Badge>
      </div>
      {verdict === "uncertain" && (
        <p className="text-sm text-muted-foreground">
          The model is unsure; compare the alternatives below and the symptoms in the library.
        </p>
      )}
    </div>
  );
}

function DiseaseInfo({ disease }: { disease: Disease }) {
  if (disease.healthy) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">No disease signs recognised</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          The model found this leaf most similar to healthy {disease.crop.toLowerCase()} leaves.
          Keep checking plants regularly; early symptoms can be subtle.
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">About {disease.name.toLowerCase()}</CardTitle>
        <CardDescription>
          {disease.pathogen} · {disease.agent_type}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p>{disease.summary}</p>
        <Section title="Typical symptoms" items={disease.symptoms} />
        <Section title="General management" items={disease.management} />
        <p>
          <Link href={`/library/${disease.slug}`} className="font-medium underline underline-offset-2">
            Full entry and sources
          </Link>
        </p>
      </CardContent>
    </Card>
  );
}

function Section({ title, items }: { title: string; items?: string[] }) {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className="mb-1 font-medium">{title}</h3>
      <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
        {items.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
    </div>
  );
}

type GradcamState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; overlay: string; ms: number }
  | { kind: "error"; message: string };

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The photo could not be read again."));
    img.src = url;
  });
}

/** The photo with the heat map blended over it (45% opacity), at the photo's own aspect ratio. */
function drawOverlay(img: HTMLImageElement, map: Float32Array): string {
  const heat = document.createElement("canvas");
  heat.width = heat.height = INPUT_SIZE;
  heat.getContext("2d")!.putImageData(new ImageData(colourise(map), INPUT_SIZE, INPUT_SIZE), 0, 0);
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0);
  ctx.globalAlpha = 0.45;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.9);
}

function GradcamPanel({ image, classIndex, label }: { image: string; classIndex: number; label: string }) {
  const [state, setState] = useState<GradcamState>({ kind: "idle" });

  async function run() {
    setState({ kind: "loading" });
    try {
      const img = await loadImage(image);
      const { map, ms } = await explain(img, classIndex);
      setState({ kind: "done", overlay: drawOverlay(img, map), ms });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : "The heat map could not be computed." });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Where did the model look?</CardTitle>
        <CardDescription>
          A Grad-CAM heat map shows which parts of the photo pushed the model towards &ldquo;{label}&rdquo;.
          It is computed on this device; the photo is not uploaded.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {state.kind === "done" ? (
          <div className="space-y-2">
            {/* eslint-disable-next-line @next/next/no-img-element -- local data URL */}
            <img src={state.overlay} alt={`Heat map over the photo showing where the model looked for ${label}`} className="w-full max-w-sm rounded-lg border" />
            <div className="flex max-w-sm items-center gap-2 text-xs text-muted-foreground" aria-hidden>
              <span>little influence</span>
              <span className="h-2 flex-1 rounded-full" style={{ background: "linear-gradient(to right, rgb(0,0,4), rgb(87,16,110), rgb(188,55,84), rgb(249,142,9), rgb(252,255,164))" }} />
              <span>strong influence</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Bright areas mattered most. On a good photo they sit on the spots or lesions, not on the background.
              A heat map explains the answer; it does not show that the answer is right. Computed in {Math.round(state.ms)} ms.
            </p>
          </div>
        ) : (
          <Button variant="outline" onClick={run} disabled={state.kind === "loading"}>
            {state.kind === "loading" ? (
              <>
                <Loader2 className="animate-spin" aria-hidden /> Computing…
              </>
            ) : (
              <>
                <Eye aria-hidden /> Show heat map
              </>
            )}
          </Button>
        )}
        {state.kind === "error" && <p className="text-sm text-destructive" role="alert">{state.message}</p>}
      </CardContent>
    </Card>
  );
}
