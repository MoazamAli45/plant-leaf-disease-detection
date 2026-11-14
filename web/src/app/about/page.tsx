import type { Metadata } from "next";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CLASSES, DISEASE_META } from "@/lib/classes";

export const metadata: Metadata = {
  title: "About",
  description: "Who made LeafDx, how it handles your photos, and the data and licences behind it.",
};

const REPO = "https://github.com/MoazamAli45/plant-leaf-disease-detection";

export default function AboutPage() {
  const healthy = CLASSES.filter((c) => c.healthy).length;
  const crops = new Set(CLASSES.map((c) => c.crop)).size;
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <header className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">About LeafDx</h1>
        <p className="text-lg text-muted-foreground">
          LeafDx is a research project by Ali Usman and Syed Moazam Ali (Department of Software
          Engineering, University of Engineering and Technology Taxila). It builds on their paper
          comparing a traditional CNN with MobileNet for plant leaf disease detection, audits the
          dataset that paper used, and tests the models on real field photos.
        </p>
      </header>

      <Block title="Your photos and privacy">
        <p>
          Analysis runs in your browser with TensorFlow.js; photos are not uploaded. The last result
          is kept only in this browser tab. The optional heat map is the one exception: it sends
          the photo to our server, only when you press the button, and the server does not store
          it.
        </p>
      </Block>

      <Block title="What it can and cannot do">
        <ul className="list-disc space-y-2 pl-5">
          <li>
            It recognises {CLASSES.length} classes ({CLASSES.length - healthy} diseases and pests,{" "}
            {healthy} healthy leaves) across {crops} crops.
          </li>
          <li>It was trained on lab photos of single leaves on plain backgrounds.</li>
          <li>
            On real field photos it is wrong most of the time (see Research). Use it to learn and to
            decide what to look up, not to decide treatment.
          </li>
          <li>Plants and diseases outside its {CLASSES.length} classes are forced into one of them.</li>
        </ul>
        <p className="font-medium">{DISEASE_META.disclaimer}</p>
      </Block>

      <Block title="Data and licences">
        <ul className="list-disc space-y-2 pl-5">
          <li>
            Training data: the Kaggle New Plant Diseases Dataset, an augmented re-release of
            PlantVillage (Hughes &amp; Salathé, 2015; Mohanty et al., 2016). The maintainers state
            CC BY-SA 3.0.
          </li>
          <li>Field test data: PlantDoc (Singh et al., 2020), CC BY 4.0.</li>
          <li>
            Disease descriptions summarise university extension publications, linked on each library
            page. {DISEASE_META.review_note}
          </li>
        </ul>
      </Block>

      <Block title="Source code and reports">
        <p>
          Code, data manifests, results and reports are on{" "}
          <a className="underline underline-offset-2" href={REPO}>
            GitHub
          </a>
          .
        </p>
      </Block>
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-muted-foreground">{children}</CardContent>
    </Card>
  );
}
