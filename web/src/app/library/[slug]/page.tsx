import { ArrowLeft, ExternalLink } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DISEASES, DISEASE_META, diseaseForSlug } from "@/lib/classes";

export async function generateStaticParams() {
  return DISEASES.map((d) => ({ slug: d.slug }));
}

export async function generateMetadata({ params }: PageProps<"/library/[slug]">): Promise<Metadata> {
  const { slug } = await params;
  const d = diseaseForSlug(slug);
  return d ? { title: `${d.name} (${d.crop})`, description: d.summary ?? `Healthy ${d.crop} leaf.` } : {};
}

export default async function DiseasePage({ params }: PageProps<"/library/[slug]">) {
  const { slug } = await params;
  const d = diseaseForSlug(slug);
  if (!d) notFound();

  return (
    <article className="mx-auto max-w-3xl space-y-6">
      <Link href="/library" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" aria-hidden /> Disease library
      </Link>
      <header className="space-y-2">
        <p className="text-sm text-muted-foreground">{d.crop}</p>
        <h1 className="text-3xl font-semibold tracking-tight">{d.name}</h1>
        {!d.healthy && (
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">{d.agent_type}</Badge>
            <Badge variant="secondary">
              <i>{d.pathogen}</i>
            </Badge>
          </div>
        )}
      </header>

      {d.healthy ? (
        <Card>
          <CardContent className="pt-6 text-muted-foreground">
            This class contains photos of healthy {d.crop.toLowerCase()} leaves. When the model
            picks it, it found no resemblance to the diseases it knows for this crop. That is not
            proof the plant is healthy: early symptoms can be subtle, and the model only knows a
            few diseases per crop.
          </CardContent>
        </Card>
      ) : (
        <>
          <p className="text-lg">{d.summary}</p>
          <InfoList title="Typical symptoms" items={d.symptoms} />
          {d.conditions && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">When and how it spreads</CardTitle>
              </CardHeader>
              <CardContent className="text-muted-foreground">{d.conditions}</CardContent>
            </Card>
          )}
          <InfoList title="General management" items={d.management} />
          <InfoList title="Can be confused with" items={d.lookalikes} />
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Sources</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                {d.sources?.map((s) => (
                  <li key={s.url}>
                    <a href={s.url} target="_blank" rel="noreferrer" className="inline-flex items-start gap-1 underline underline-offset-2">
                      {s.title}
                      <ExternalLink className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                      <span className="sr-only">(opens in a new tab)</span>
                    </a>
                    <span className="text-muted-foreground"> · {s.publisher}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </>
      )}

      <Alert>
        <AlertDescription>
          {DISEASE_META.disclaimer} {DISEASE_META.review_note}
        </AlertDescription>
      </Alert>
    </article>
  );
}

function InfoList({ title, items }: { title: string; items?: string[] }) {
  if (!items?.length) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
          {items.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
