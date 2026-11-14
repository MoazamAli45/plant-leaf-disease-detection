import type { Metadata } from "next";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { DISEASE_META, byCrop } from "@/lib/classes";

export const metadata: Metadata = {
  title: "Disease library",
  description: "The 38 leaf conditions LeafDx recognises, with symptoms, management and sources.",
};

export default function LibraryPage() {
  const groups = byCrop();
  const total = groups.reduce((n, g) => n + g.diseases.length, 0);
  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Disease library</h1>
        <p className="max-w-prose text-muted-foreground">
          The {total} classes the model recognises, across {groups.length} crops. Each disease
          entry summarises university extension sources, which are linked on its page.
        </p>
        <p className="max-w-prose text-sm text-muted-foreground">{DISEASE_META.review_note}</p>
      </header>
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {groups.map((g) => (
          <section key={g.crop} aria-labelledby={`crop-${g.crop}`} className="rounded-xl border p-4">
            <h2 id={`crop-${g.crop}`} className="mb-3 font-semibold">
              {g.crop}
            </h2>
            <ul className="space-y-2">
              {g.diseases.map((d) => (
                <li key={d.slug} className="flex items-center justify-between gap-2">
                  <Link href={`/library/${d.slug}`} className="text-sm underline-offset-2 hover:underline">
                    {d.name}
                  </Link>
                  {d.healthy ? (
                    <Badge variant="secondary">healthy</Badge>
                  ) : (
                    <Badge variant="outline">{d.agent_type?.split(" ")[0]}</Badge>
                  )}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
