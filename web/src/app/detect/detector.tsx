"use client";

import { Camera, ImagePlus, Loader2, RotateCcw, Upload, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { classify, loadModel } from "@/lib/model";
import { saveAnalysis } from "@/lib/storage";
import { plantPixelShare, topK, verdict } from "@/lib/verdict";

type Status =
  | { kind: "idle" }
  | { kind: "camera" }
  | { kind: "ready"; url: string }
  | { kind: "analysing"; url: string }
  | { kind: "error"; message: string; url?: string };

const MAX_BYTES = 15 * 1024 * 1024;

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("This file could not be read as an image."));
    img.src = url;
  });
}

/** Downscaled JPEG for the result page, and a tiny RGBA sample for the leaf check. */
function summarise(img: HTMLImageElement): { dataUrl: string; sample: Uint8ClampedArray } {
  const scale = Math.min(1, 640 / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(img.naturalWidth * scale);
  c.height = Math.round(img.naturalHeight * scale);
  c.getContext("2d")!.drawImage(img, 0, 0, c.width, c.height);
  const s = document.createElement("canvas");
  s.width = s.height = 64;
  const ctx = s.getContext("2d")!;
  ctx.drawImage(img, 0, 0, 64, 64);
  return { dataUrl: c.toDataURL("image/jpeg", 0.85), sample: ctx.getImageData(0, 0, 64, 64).data };
}

export function Detector() {
  const router = useRouter();
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [modelProgress, setModelProgress] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);

  // start downloading the model as soon as the page opens
  useEffect(() => {
    let cancelled = false;
    loadModel((f) => !cancelled && setModelProgress(f))
      .then(() => !cancelled && setModelReady(true))
      .catch(() =>
        !cancelled &&
        setModelError(
          "The model could not be loaded. Check your connection; after one successful visit it also works offline.",
        ),
      );
    return () => {
      cancelled = true;
    };
  }, []);

  const stopCamera = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);
  useEffect(() => stopCamera, [stopCamera]);

  function acceptFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus({ kind: "error", message: "Please choose an image file (JPEG, PNG or WebP)." });
      return;
    }
    if (file.size > MAX_BYTES) {
      setStatus({ kind: "error", message: "That image is larger than 15 MB. Please use a smaller photo." });
      return;
    }
    setStatus({ kind: "ready", url: URL.createObjectURL(file) });
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      fileInput.current?.click(); // phones without getUserMedia still offer the camera here
      return;
    }
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      setStatus({ kind: "camera" });
      requestAnimationFrame(() => {
        if (video.current && stream.current) {
          video.current.srcObject = stream.current;
          void video.current.play();
        }
      });
    } catch (e) {
      const denied = e instanceof DOMException && e.name === "NotAllowedError";
      setStatus({
        kind: "error",
        message: denied
          ? "Camera permission was denied. Allow camera access in your browser settings, or upload a photo instead."
          : "No camera could be opened on this device. Upload a photo instead.",
      });
    }
  }

  function capture() {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    stopCamera();
    c.toBlob((blob) => blob && setStatus({ kind: "ready", url: URL.createObjectURL(blob) }), "image/jpeg", 0.92);
  }

  async function analyse(url: string) {
    setStatus({ kind: "analysing", url });
    try {
      const img = await loadImage(url);
      const { probs, backend, ms } = await classify(img);
      const { dataUrl, sample } = summarise(img);
      const share = plantPixelShare(sample);
      const top = topK(probs, 3);
      saveAnalysis({
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        image: dataUrl,
        top,
        verdict: verdict(top, share),
        plantPixelShare: share,
        backend,
        ms,
      });
      router.push("/result");
    } catch (e) {
      setStatus({
        kind: "error",
        url,
        message: e instanceof Error ? e.message : "Something went wrong while analysing the photo.",
      });
    }
  }

  function reset() {
    stopCamera();
    setStatus({ kind: "idle" });
  }

  const url = "url" in status ? status.url : undefined;

  return (
    <div className="space-y-6">
      {!modelReady && !modelError && (
        <div className="space-y-2" aria-live="polite">
          <p className="text-sm text-muted-foreground">
            Loading the model ({Math.round(modelProgress * 100)}%). This happens once; afterwards it
            is cached for offline use.
          </p>
          <Progress value={modelProgress * 100} aria-label="Model download progress" />
        </div>
      )}
      {modelError && (
        <Alert variant="destructive">
          <AlertTitle>Model not available</AlertTitle>
          <AlertDescription>{modelError}</AlertDescription>
        </Alert>
      )}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        aria-hidden
        tabIndex={-1}
        onChange={(e) => acceptFile(e.target.files?.[0])}
      />

      {status.kind === "camera" ? (
        <Card>
          <CardContent className="space-y-4 pt-6">
            <video
              ref={video}
              playsInline
              muted
              className="aspect-[4/3] w-full rounded-lg bg-black object-cover"
              aria-label="Camera preview"
            />
            <div className="flex flex-wrap gap-3">
              <Button onClick={capture}>
                <Camera aria-hidden /> Take photo
              </Button>
              <Button variant="outline" onClick={reset}>
                <X aria-hidden /> Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : url ? (
        <Card>
          <CardContent className="space-y-4 pt-6">
            {/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
            <img
              src={url}
              alt="The photo you selected"
              className="mx-auto max-h-[60vh] rounded-lg object-contain"
            />
            <div className="flex flex-wrap gap-3">
              <Button
                onClick={() => analyse(url)}
                disabled={status.kind === "analysing" || !modelReady}
              >
                {status.kind === "analysing" ? (
                  <>
                    <Loader2 className="animate-spin" aria-hidden /> Analysing…
                  </>
                ) : (
                  "Analyse this leaf"
                )}
              </Button>
              <Button variant="outline" onClick={reset} disabled={status.kind === "analysing"}>
                <RotateCcw aria-hidden /> Choose another photo
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            acceptFile(e.dataTransfer.files?.[0]);
          }}
          className={`flex flex-col items-center gap-4 rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
            dragging ? "border-primary bg-primary/5" : "border-border"
          }`}
        >
          <ImagePlus className="size-10 text-muted-foreground" aria-hidden />
          <div>
            <p className="font-medium">Drop a photo of one leaf here</p>
            <p className="text-sm text-muted-foreground">JPEG, PNG or WebP, up to 15 MB</p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            <Button onClick={() => fileInput.current?.click()}>
              <Upload aria-hidden /> Upload a photo
            </Button>
            <Button variant="outline" onClick={startCamera}>
              <Camera aria-hidden /> Use camera
            </Button>
          </div>
        </div>
      )}

      {status.kind === "error" && (
        <Alert variant="destructive" role="alert">
          <AlertTitle>Something went wrong</AlertTitle>
          <AlertDescription>{status.message}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
