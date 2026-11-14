import "server-only";

import fs from "node:fs";
import path from "node:path";

/** Size of the deployed TF.js model (model.json + weight shards), read at build time. */
export function modelSizeMb(): string {
  const dir = path.join(process.cwd(), "public", "model");
  const bytes = fs
    .readdirSync(dir)
    .reduce((sum, f) => sum + fs.statSync(path.join(dir, f)).size, 0);
  return (bytes / 1e6).toFixed(1);
}
