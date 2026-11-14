// Copies the TF.js WebAssembly binaries into public/tfjs-wasm so the browser (and the service
// worker, for offline use) can load them from this site. Runs before dev and build.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const src = path.join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
const dest = path.join(root, "public", "tfjs-wasm");
fs.mkdirSync(dest, { recursive: true });
for (const f of fs.readdirSync(src).filter((f) => f.endsWith(".wasm"))) {
  fs.copyFileSync(path.join(src, f), path.join(dest, f));
}
console.log("copied TF.js wasm binaries to public/tfjs-wasm");
