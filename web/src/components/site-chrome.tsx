"use client";

import { Leaf } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

import { ThemeToggle } from "@/components/theme";

const NAV = [
  { href: "/detect", label: "Detect" },
  { href: "/library", label: "Disease library" },
  { href: "/research", label: "Research" },
  { href: "/about", label: "About" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-40">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-50 focus:rounded-md focus:bg-background focus:px-3 focus:py-2 focus:ring-2 focus:ring-ring"
      >
        Skip to content
      </a>
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Leaf className="size-5 text-primary" aria-hidden />
          <span>LeafDx</span>
        </Link>
        <nav aria-label="Main" className="order-3 w-full sm:order-none sm:w-auto">
          <ul className="flex flex-wrap gap-1 text-sm">
            {NAV.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`rounded-md px-3 py-1.5 transition-colors hover:bg-muted ${
                      active ? "bg-muted font-medium text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t">
      <div className="mx-auto max-w-6xl space-y-2 px-4 py-8 text-sm text-muted-foreground">
        <p>
          <strong className="text-foreground">Not agronomic advice.</strong> LeafDx is a research
          project. Its accuracy on real field photos is low (see{" "}
          <Link href="/research" className="underline underline-offset-2">
            Research
          </Link>
          ). Confirm any diagnosis with a local extension office or plant diagnostic clinic.
        </p>
        <p>
          Trained on PlantVillage (CC BY-SA 3.0) and evaluated on PlantDoc (CC BY 4.0). Photos are
          analysed in your browser and are not uploaded unless you ask for a heat map.
        </p>
      </div>
    </footer>
  );
}

/** Registers the service worker (production only) so the app and model work offline. */
export function ServiceWorker() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // offline support is optional; the app works without it
    });
  }, []);
  return null;
}
