"""Re-audit with the corrected file-name parser (``_newPixel<n>`` recognised).

    python -m leafdx.data.recheck

Split v1 was built by a parser that missed the ``_newPixel25`` augmentation, so those copies
were not grouped with their source photos. This script recomputes, from the prepare cache:
- the paper 80/20 split's leakage with the corrected parsing;
- the residual leakage of clean split v1 (val/test images whose source photo or physical leaf
  also appears in v1's train split).

Outputs: ``docs/figures/data_audit/recheck_newpixel.json`` and
``manifests/clean_split/{val,test}_links_corrected.csv`` (one row per image, with flags). The
latter defines the **leak-free subset** used to report Phase 3 test results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from leafdx.data.filenames import parse_npd_filename, resolve_leaf_id


def corrected_ids(df: pd.DataFrame, leaf_map: dict[str, list[str]]) -> pd.DataFrame:
    names = df["path"].str.rsplit("/", n=1).str[-1]
    parsed = [parse_npd_filename(n, newpixel=True) for n in names]
    out = df.copy()
    out["source_id_v2"] = [
        f"{lab}/{p.source_stem}" for lab, p in zip(df["label"], parsed, strict=True)
    ]
    out["leaf_id_v2"] = [
        resolve_leaf_id(p.source_stem, lab, leaf_map)
        for lab, p in zip(df["label"], parsed, strict=True)
    ]
    out["aug_type_v2"] = [p.aug_type for p in parsed]
    return out


def links(df: pd.DataFrame, split_col: str, eval_split: str) -> pd.DataFrame:
    tr = df[df[split_col] == "train"]
    ev = df[df[split_col] == eval_split]
    src = ev["source_id_v2"].isin(set(tr["source_id_v2"]))
    leaf = ev["leaf_id_v2"].notna() & ev["leaf_id_v2"].isin(set(tr["leaf_id_v2"].dropna()))
    sha = ev["sha256"].isin(set(tr["sha256"]))
    return pd.DataFrame(
        {
            "path": ev["path"],
            "linked_source": src,
            "linked_leaf": leaf,
            "linked_exact": sha,
            "linked_any": src | leaf | sha,
        }
    )


def summarise(lk: pd.DataFrame) -> dict[str, Any]:
    n = len(lk)
    return {
        "n": n,
        **{
            f"n_{c}": int(lk[c].sum())
            for c in ("linked_source", "linked_leaf", "linked_exact", "linked_any")
        },
        **{
            f"frac_{c}": float(lk[c].mean())
            for c in ("linked_source", "linked_leaf", "linked_exact", "linked_any")
        },
    }


def run(data_dir: Path, manifest_dir: Path, report_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(
        data_dir / "cache" / "npd_full.csv", keep_default_na=False, na_values=[""], low_memory=False
    )
    leaf_map = json.loads((data_dir / "raw" / "plantvillage" / "leaf-map.json").read_text())
    df = corrected_ids(df, leaf_map)
    newpixel = df["aug_type_v2"].str.contains("newPixel")
    result: dict[str, Any] = {
        "n_newpixel_images": int(newpixel.sum()),
        "n_newpixel_with_leaf_id_after_fix": int(df.loc[newpixel, "leaf_id_v2"].notna().sum()),
        "paper_split_val": summarise(links(df, "paper_split", "val")),
    }
    for split in ("val", "test"):
        lk = links(df, "clean_split", split)
        lk.sort_values("path").to_csv(
            manifest_dir / "clean_split" / f"{split}_links_corrected.csv",
            index=False,
            lineterminator="\n",
        )
        result[f"clean_split_v1_{split}"] = summarise(lk)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "recheck_newpixel.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    p.add_argument("--report-dir", type=Path, default=Path("../docs/figures/data_audit"))
    args = p.parse_args(argv)
    print(json.dumps(run(args.data_dir, args.manifest_dir, args.report_dir), indent=2))


if __name__ == "__main__":
    main()
