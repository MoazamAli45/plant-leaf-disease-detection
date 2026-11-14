import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "LeafDx: plant leaf disease detection",
    short_name: "LeafDx",
    description: "Check a plant leaf for disease, privately and offline, in your browser.",
    start_url: "/detect",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#2f6b45",
    categories: ["education", "utilities"],
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
