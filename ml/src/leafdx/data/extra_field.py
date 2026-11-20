"""Extra field-photo datasets for Phase 7 (protocol: docs/field2.md).

    python -m leafdx.data.extra_field cds          # CD&S corn photos (OSF) -> data/raw/cds/
    python -m leafdx.data.extra_field fieldplant   # FieldPlant leaf crops -> data/raw/fieldplant/

CD&S (Ahmad et al., 2021; https://osf.io/s6ru5) states no licence, so it is used only as an
independent **test** set and never for training; its images are not redistributed.
FieldPlant (Moupojou et al., 2023) is CC BY 4.0 (Roboflow export on Kaggle, ``bloox2/fieldplant``);
each annotated leaf box becomes one classification image.

Both write a manifest CSV (path, label, source_image, ...) and add an entry to
``data/raw/PROVENANCE.json`` with the source, licence and per-file SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CDS_NODE = "s6ru5"
# Dataset_Original/{train,test}/{gls,nlb}: the two CD&S classes that exist in the 38-class model
# (nls, Northern Leaf Spot, has no PlantVillage counterpart)
CDS_FOLDERS = {
    ("test", "gls"): "60be2964cb2a5e0501690f5e",
    ("test", "nlb"): "60be297e9096b705086437f4",
    ("train", "gls"): "60be29b19096b7050463f1b6",
    ("train", "nlb"): "60be29c4cb2a5e04f868b834",
}
CDS_LABELS = {
    "gls": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "nlb": "Corn_(maize)___Northern_Leaf_Blight",
}


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _provenance(data_root: Path, key: str, entry: dict) -> None:
    f = data_root / "raw" / "PROVENANCE.json"
    prov = json.loads(f.read_text()) if f.exists() else {}
    prov[key] = entry
    f.write_text(json.dumps(prov, indent=1) + "\n")


def _osf_files(folder_id: str) -> list[dict]:
    url = f"https://api.osf.io/v2/nodes/{CDS_NODE}/files/osfstorage/{folder_id}/"
    out = []
    while url:
        d = requests.get(url, timeout=60).json()
        out += [c for c in d["data"] if c["attributes"]["kind"] == "file"]
        url = d["links"].get("next")
    return out


def _fetch(url: str, dest: Path, sha256: str | None, tries: int = 12) -> str:
    """Download with checksum check. OSF rate-limits (HTTP 429): wait as long as it asks
    (Retry-After), otherwise back off exponentially, up to 10 minutes."""
    if dest.exists() and (sha256 is None or _sha256(dest.read_bytes()) == sha256):
        return _sha256(dest.read_bytes())
    for i in range(tries):
        wait = min(600.0, 5.0 * 2**i)
        try:
            r = requests.get(url, timeout=900)
            if r.status_code == 429:
                wait = max(wait, float(r.headers.get("Retry-After", 60)))
            r.raise_for_status()
            got = _sha256(r.content)
            if sha256 and got != sha256:
                raise ValueError(f"checksum mismatch for {dest.name}")
            dest.write_bytes(r.content)
            return got
        except (requests.RequestException, ValueError):
            if i == tries - 1:
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")


def cds(data_root: Path, workers: int = 8) -> Path:
    out = data_root / "raw" / "cds"
    rows = []
    for (split, cls), folder in CDS_FOLDERS.items():
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        files = _osf_files(folder)

        def get(f: dict, d: Path = d) -> tuple[str, str]:
            a = f["attributes"]
            sha = (a.get("extra") or {}).get("hashes", {}).get("sha256")
            return a["name"], _fetch(f["links"]["download"], d / a["name"], sha)

        with ThreadPoolExecutor(workers) as ex:
            for name, sha in ex.map(get, files):
                rows.append(
                    {
                        "path": f"raw/cds/{split}/{cls}/{name}",
                        "label": CDS_LABELS[cls],
                        "cds_split": split,
                        "cds_class": cls,
                        "sha256": sha,
                    }
                )
        print(split, cls, len(files))
    df = pd.DataFrame(rows).sort_values("path")
    man = Path("manifests/cds")
    man.mkdir(parents=True, exist_ok=True)
    df.to_csv(man / "all.csv", index=False)
    _provenance(
        data_root,
        "cds",
        {
            "source": {
                "url": f"https://osf.io/{CDS_NODE}/",
                "folders": "Dataset_Original/{train,test}/{gls,nlb}",
                "license": "none stated on OSF (checked 2026-11-18); used for evaluation only",
                "citation": "Ahmad et al., CD&S Dataset, arXiv:2110.12084, 2021",
            },
            "n_files": len(df),
            "manifest": "manifests/cds/all.csv (per-file sha256)",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return man / "all.csv"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("dataset", choices=["cds", "fieldplant"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (OSF is slow)")
    a = p.parse_args(argv)
    if a.dataset == "cds":
        print("wrote", cds(a.data_root, a.workers))
    else:
        from leafdx.data.fieldplant import prepare

        print("wrote", prepare(a.data_root))


if __name__ == "__main__":
    main()
