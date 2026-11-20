// LeafDx service worker: makes the app shell and the model available offline.
// - Pages: network first, falling back to the cached copy (so updates arrive when online).
// - Model, wasm and built assets: cache first (they are immutable or versioned).
// Bump VERSION whenever the model files change, or browsers keep serving the old model.
const VERSION = "leafdx-339eae100fa4";
const PRECACHE = [
  "/",
  "/detect",
  "/result",
  "/library",
  "/research",
  "/about",
  "/model/group1-shard1of2.bin",
  "/model/group1-shard2of2.bin",
  "/model/leafdx.json",
  "/model/model.json",
  "/tfjs-wasm/tfjs-backend-wasm.wasm",
  "/tfjs-wasm/tfjs-backend-wasm-simd.wasm",
  "/tfjs-wasm/tfjs-backend-wasm-threaded-simd.wasm",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) =>
      // add files one by one so a single missing file does not abort installation
      Promise.all(PRECACHE.map((url) => cache.add(url).catch(() => undefined))),
    ),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function cacheFirst(request) {
  return caches.match(request).then(
    (hit) =>
      hit ||
      fetch(request).then((res) => {
        if (res.ok) caches.open(VERSION).then((c) => c.put(request, res.clone()));
        return res;
      }),
  );
}

function networkFirst(request) {
  return fetch(request)
    .then((res) => {
      if (res.ok) caches.open(VERSION).then((c) => c.put(request, res.clone()));
      return res;
    })
    .catch(() => caches.match(request).then((hit) => hit || caches.match("/detect")));
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  if (
    url.pathname.startsWith("/model/") ||
    url.pathname.startsWith("/tfjs-wasm/") ||
    url.pathname.startsWith("/_next/static/") ||
    url.pathname.startsWith("/icons/")
  ) {
    event.respondWith(cacheFirst(request));
  } else if (request.mode === "navigate" || request.headers.get("accept")?.includes("text/html")) {
    event.respondWith(networkFirst(request));
  } else {
    event.respondWith(networkFirst(request));
  }
});
