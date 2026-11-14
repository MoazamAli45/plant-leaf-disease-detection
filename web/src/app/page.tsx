import { ArrowRight, FlaskConical, ShieldCheck, WifiOff } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CLASSES } from "@/lib/classes";
import { modelSizeMb } from "@/lib/model-info";
import { headline, pct } from "@/lib/research";

export default function Home() {
  const h = headline();
  const crops = new Set(CLASSES.map((c) => c.crop)).size;
  const size = modelSizeMb();
  return (
    <div className="space-y-16">
      <section className="grid items-center gap-10 py-6 md:grid-cols-[1.2fr_1fr]">
        <div className="space-y-6">
          <p className="text-sm font-medium text-primary">Research project · open source</p>
          <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
            Check a plant leaf for disease, privately in your browser
          </h1>
          <p className="max-w-prose text-lg text-muted-foreground">
            LeafDx recognises {CLASSES.length} leaf conditions across {crops} crops from a single
            photo. The model runs on your device: no photo is uploaded, and it keeps working
            offline.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button asChild size="lg">
              <Link href="/detect">
                Check a leaf <ArrowRight aria-hidden />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/research">How well does it work?</Link>
            </Button>
          </div>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>What the evaluation found</CardTitle>
            <CardDescription>
              {"Deployed model: MobileNetV1, mean of 3 training runs."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Stat
              value={`${h.labLeakFree.mean.toFixed(1)}%`}
              label="accuracy on held-out lab photos (no overlap with training)"
            />
            <Stat
              value={`${h.field.mean.toFixed(1)}%`}
              label={`accuracy on ${h.fieldImages.toLocaleString()} real field photos (PlantDoc)`}
              tone="warn"
            />
            <Stat
              value={pct(h.paperValLinked)}
              label="of the original paper's validation images had a copy or same-leaf photo in training"
            />
            <p className="text-sm text-muted-foreground">
              Lab accuracy does not carry over to field photos. Treat every result as a hint, not
              a diagnosis.
            </p>
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="how" className="space-y-6">
        <h2 id="how" className="text-2xl font-semibold tracking-tight">
          How it works
        </h2>
        <div className="grid gap-4 md:grid-cols-3">
          <Feature
            icon={<ShieldCheck aria-hidden className="size-5 text-primary" />}
            title="Private by default"
            text={`The model (${size} MB) is downloaded once and runs in your browser with TensorFlow.js. A heat map of where the model looked is optional and asks before sending the photo.`}
          />
          <Feature
            icon={<WifiOff aria-hidden className="size-5 text-primary" />}
            title="Works offline"
            text="Install it to your home screen. After the first visit the app and model are cached, so you can check leaves without a connection."
          />
          <Feature
            icon={<FlaskConical aria-hidden className="size-5 text-primary" />}
            title="Honest about its limits"
            text="Low-confidence photos get a 'not sure' answer instead of a guess. The research page shows where the model fails, not only where it succeeds."
          />
        </div>
      </section>
    </div>
  );
}

function Stat({ value, label, tone }: { value: string; label: string; tone?: "warn" }) {
  return (
    <div className="flex items-baseline gap-3">
      <span
        className={`text-3xl font-semibold tabular-nums ${tone === "warn" ? "text-orange-700 dark:text-orange-400" : ""}`}
      >
        {value}
      </span>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  );
}

function Feature({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          {icon}
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="text-sm text-muted-foreground">{text}</CardContent>
    </Card>
  );
}
