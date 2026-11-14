"use client";

import type { Prediction, Verdict } from "@/lib/verdict";

/** The last analysis, handed from /detect to /result. Kept only in this browser tab. */
export type Analysis = {
  id: string;
  createdAt: string;
  image: string; // data URL, downscaled
  top: Prediction[];
  verdict: Verdict;
  plantPixelShare: number;
  backend: string;
  ms: number;
};

const KEY = "leafdx:last-analysis";

export function saveAnalysis(a: Analysis): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(a));
  } catch {
    // private mode or full storage: the result page will ask to analyse again
  }
}

/** The stored JSON string (stable identity while unchanged, for useSyncExternalStore). */
export function readAnalysisRaw(): string | null {
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function parseAnalysis(raw: string | null): Analysis | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Analysis;
  } catch {
    return null;
  }
}
