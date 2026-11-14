import classes from "@/data/classes.json";
import diseases from "@/data/diseases.json";

export type ClassInfo = {
  index: number;
  class: string;
  crop: string;
  name: string;
  healthy: boolean;
};

export type Source = { title: string; publisher: string; url: string };

export type Disease = {
  class: string;
  slug: string;
  crop: string;
  name: string;
  healthy: boolean;
  agent_type?: string;
  pathogen?: string;
  summary?: string;
  symptoms?: string[];
  conditions?: string;
  management?: string[];
  lookalikes?: string[];
  sources?: Source[];
};

export const CLASSES: ClassInfo[] = classes;
export const DISEASES: Disease[] = diseases.classes;
export const DISEASE_META = diseases.meta;

const byClass = new Map(DISEASES.map((d) => [d.class, d]));
const bySlug = new Map(DISEASES.map((d) => [d.slug, d]));

export function diseaseForClass(className: string): Disease | undefined {
  return byClass.get(className);
}

export function diseaseForSlug(slug: string): Disease | undefined {
  return bySlug.get(slug);
}

export function diseaseForIndex(index: number): Disease | undefined {
  const c = CLASSES[index];
  return c ? byClass.get(c.class) : undefined;
}

/** Crops in display order with their classes. */
export function byCrop(): { crop: string; diseases: Disease[] }[] {
  const groups = new Map<string, Disease[]>();
  for (const d of DISEASES) {
    if (!groups.has(d.crop)) groups.set(d.crop, []);
    groups.get(d.crop)!.push(d);
  }
  return [...groups.entries()].map(([crop, list]) => ({
    crop,
    diseases: list.sort((a, b) => Number(a.healthy) - Number(b.healthy) || a.name.localeCompare(b.name)),
  }));
}
