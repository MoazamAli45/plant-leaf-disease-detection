import type { Metadata } from "next";

import { headline } from "@/lib/research";

import { ResultView } from "./result-view";

export const metadata: Metadata = {
  title: "Result",
  description: "The model's answer for your leaf photo, with confidence and disease information.",
};

export default function ResultPage() {
  const field = `${headline().field.mean.toFixed(0)}%`;
  return <ResultView fieldAccuracy={field} />;
}
