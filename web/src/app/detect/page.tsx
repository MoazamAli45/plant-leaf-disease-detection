import type { Metadata } from "next";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import { DISEASES } from "@/lib/classes";

import { Detector } from "./detector";

export const metadata: Metadata = {
  title: "Detect",
  description: "Upload or take a photo of a plant leaf and check it for disease in your browser.",
};

export default function DetectPage() {
  const crops = [...new Set(DISEASES.map((d) => d.crop.toLowerCase()))];
  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_18rem]">
      <section className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">Check a leaf</h1>
        <p className="text-muted-foreground">
          The photo is analysed on this device and is not uploaded.
        </p>
        <Detector />
      </section>
      <aside aria-labelledby="tips">
        <Card>
          <CardHeader>
            <CardTitle id="tips" className="text-base">
              Tips for a useful result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <ul className="list-disc space-y-2 pl-4">
              <li>One leaf, filling most of the frame.</li>
              <li>A plain background and even, indirect light.</li>
              <li>Show the side with the clearest symptoms.</li>
            </ul>
            <p>
              The model was trained on lab photos of single leaves. It is much less reliable on
              busy field photos with several leaves, soil or hands in the frame.
            </p>
            <p>
              It knows {crops.length} crops: {crops.join(", ")}. Other plants will be forced into
              one of these classes.
            </p>
          </CardContent>
        </Card>
      </aside>
    </div>
  );
}
