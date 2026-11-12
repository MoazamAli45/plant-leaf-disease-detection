"""Download the raw datasets into a canonical layout.

    data/raw/npd/{train,valid}/<class>/<file>         New Plant Diseases Dataset (Kaggle)
    data/raw/plantdoc/{train,test}/<class>/<file>     PlantDoc classification set (GitHub)
    data/raw/plantvillage/leaf-map.json               upstream leaf grouping metadata
    data/raw/PROVENANCE.json                          what was downloaded, from where, checksums

Usage:
    python -m leafdx.data.download --data-dir data            # everything
    python -m leafdx.data.download --only plantdoc leafmap    # no Kaggle account needed

The Kaggle download needs an API token (https://www.kaggle.com/settings -> API), either in
~/.kaggle/kaggle.json (legacy) or the KAGGLE_API_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from tqdm import tqdm

from leafdx.data import sources
from leafdx.data.classes import PLANTVILLAGE_CLASSES

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_url(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename). Skips if ``dest`` exists."""
    if dest.exists():
        log.info("already downloaded: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            tmp.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(_CHUNK):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


@dataclass
class ExtractStats:
    written: int = 0
    duplicate_identical: int = 0
    duplicate_conflicting: list[str] = field(default_factory=list)
    skipped_non_image: int = 0
    skipped_outside_tree: list[str] = field(default_factory=list)
    unknown_classes: dict[str, int] = field(default_factory=dict)
    # names that differ only by case would overwrite each other on macOS/Windows file systems
    case_renamed: dict[str, str] = field(default_factory=dict)


def _safe_component(part: str) -> bool:
    return part not in {"", ".", ".."} and "/" not in part and "\\" not in part


def locate_in_tree(
    member: str, split_names: Iterable[str], is_class: Callable[[str], bool]
) -> tuple[str, str, str] | None:
    """Find ``.../<split>/<class>/<file>`` inside an archive member path.

    Returns (split, class, filename) or None. The *last* matching split component wins, so
    nested wrapper folders (e.g. ``Foo/Foo/train/...``) are handled.
    """
    parts = PurePosixPath(member).parts
    splits = set(split_names)
    for i in range(len(parts) - 3, -1, -1):
        if parts[i] in splits and len(parts) == i + 3 and is_class(parts[i + 1]):
            split, cls, fname = parts[i], parts[i + 1], parts[i + 2]
            if all(_safe_component(p) for p in (split, cls, fname)):
                return split, cls, fname
    return None


def extract_class_tree(
    archive: Path,
    out_dir: Path,
    split_names: Iterable[str],
    known_classes: set[str] | None = None,
) -> ExtractStats:
    """Extract image files laid out as <split>/<class>/<file> into ``out_dir``.

    If ``known_classes`` is given, folders with other names are counted in
    ``unknown_classes`` and not extracted. The same relative path appearing twice in the archive
    (some Kaggle zips contain duplicated wrapper folders) is kept once if byte-identical and
    reported otherwise.
    """
    stats = ExtractStats()
    split_names = list(split_names)
    seen: dict[str, str] = {}  # relative path -> sha256
    taken_ci: set[str] = set()  # case-folded paths already written
    # sorted so that renaming of case collisions is deterministic
    with zipfile.ZipFile(archive) as zf:
        members = sorted(zf.infolist(), key=lambda m: m.filename)
        for info in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            if info.is_dir():
                continue
            if PurePosixPath(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                stats.skipped_non_image += 1
                continue
            loc = locate_in_tree(info.filename, split_names, lambda _c: True)
            if loc is None:
                stats.skipped_outside_tree.append(info.filename)
                continue
            split, cls, fname = loc
            if known_classes is not None and cls not in known_classes:
                stats.unknown_classes[cls] = stats.unknown_classes.get(cls, 0) + 1
                continue
            rel = f"{split}/{cls}/{fname}"
            data = zf.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if rel in seen:
                if seen[rel] == digest:
                    stats.duplicate_identical += 1
                else:
                    stats.duplicate_conflicting.append(info.filename)
                continue
            seen[rel] = digest
            if rel.casefold() in taken_ci:
                stem, suffix = PurePosixPath(fname).stem, PurePosixPath(fname).suffix
                n = 2
                while f"{split}/{cls}/{stem}__case{n}{suffix}".casefold() in taken_ci:
                    n += 1
                fname = f"{stem}__case{n}{suffix}"
                stats.case_renamed[rel] = f"{split}/{cls}/{fname}"
                rel = f"{split}/{cls}/{fname}"
            taken_ci.add(rel.casefold())
            target = out_dir / split / cls / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            stats.written += 1
    return stats


def _complete_marker(out_dir: Path) -> Path:
    return out_dir / ".extract-complete.json"


def _extract_once(
    archive: Path, out_dir: Path, splits: list[str], known: set[str] | None
) -> dict[str, Any]:
    """Extract unless a marker shows this exact archive was already extracted."""
    digest = sha256_file(archive)
    marker = _complete_marker(out_dir)
    if marker.exists():
        info = json.loads(marker.read_text())
        if info.get("archive_sha256") == digest:
            log.info("already extracted: %s", out_dir)
            return info
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = extract_class_tree(archive, out_dir, splits, known)
    if stats.duplicate_conflicting:
        log.warning("%d conflicting duplicate paths", len(stats.duplicate_conflicting))
    info = {"archive": archive.name, "archive_sha256": digest, **asdict(stats)}
    # keep the marker small: lists are summarised
    info["skipped_outside_tree"] = {
        "count": len(stats.skipped_outside_tree),
        "examples": stats.skipped_outside_tree[:10],
    }
    marker.write_text(json.dumps(info, indent=2))
    return info


def kaggle_metadata(ref: str, retries: int = 3) -> dict[str, Any]:
    """Dataset metadata (version, license, size).

    Tries the public endpoint (no auth) with retries, since it intermittently returns 404,
    then falls back to the authenticated client. ``source`` records which one answered.
    """
    keys = ["ref", "title", "licenseName", "currentVersionNumber", "lastUpdated", "totalBytes"]
    for attempt in range(retries):
        try:
            r = requests.get(sources.KAGGLE_VIEW_API.format(ref=ref), timeout=30)
            r.raise_for_status()
            d = r.json()
            return {**{k: d.get(k) for k in keys}, "source": "public_api"}
        except (requests.RequestException, ValueError) as e:
            log.warning("Kaggle public metadata attempt %d failed: %s", attempt + 1, e)
            time.sleep(2**attempt)
    api = _kaggle_api()
    owner, slug = ref.split("/", 1)
    match = [d for d in api.dataset_list(search=slug, user=owner) if d.ref == ref]
    if not match:
        raise RuntimeError(f"Kaggle dataset {ref} not found")
    d = match[0]
    return {
        "ref": d.ref,
        "title": d.title,
        "licenseName": d.license_name,
        "currentVersionNumber": d.current_version_number,
        "lastUpdated": str(d.last_updated),
        "totalBytes": d.total_bytes,
        "source": "authenticated_api",
    }


def _kaggle_api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: authenticates on import
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install -e '.[download]' to get the kaggle client") from e
    api = KaggleApi()
    api.authenticate()
    return api


def download_npd(data_dir: Path, allow_version_change: bool = False) -> dict[str, Any]:
    src = sources.NEW_PLANT_DISEASES
    meta = kaggle_metadata(src.ref)
    if meta["currentVersionNumber"] != src.expected_version and not allow_version_change:
        raise RuntimeError(
            f"Kaggle dataset {src.ref} is at version {meta['currentVersionNumber']}, expected "
            f"{src.expected_version}. Re-run with --allow-version-change and re-audit."
        )
    downloads = data_dir / "downloads"
    archive = downloads / "new-plant-diseases-dataset.zip"
    # the Kaggle client writes to the final name and marks unfinished downloads with a
    # sidecar file; treat those as missing so an interrupted download is resumed, not used
    partial = archive.with_name(archive.name + ".kaggle-partial")
    if not archive.exists() or partial.exists():
        api = _kaggle_api()
        downloads.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(src.ref, path=str(downloads), unzip=False, quiet=False)
    extract = _extract_once(
        archive, data_dir / "raw" / "npd", ["train", "valid"], set(PLANTVILLAGE_CLASSES)
    )
    return {"source": asdict(src), "kaggle_metadata": meta, "extract": extract}


def download_plantdoc(data_dir: Path) -> dict[str, Any]:
    src = sources.PLANTDOC
    archive = download_url(src.url, data_dir / "downloads" / f"plantdoc-{src.revision[:12]}.zip")
    extract = _extract_once(archive, data_dir / "raw" / "plantdoc", ["train", "test"], None)
    return {"source": asdict(src), "extract": extract}


def download_leaf_map(data_dir: Path) -> dict[str, Any]:
    src = sources.PLANTVILLAGE_LEAF_MAP
    dest = download_url(src.url, data_dir / "raw" / "plantvillage" / "leaf-map.json")
    return {"source": asdict(src), "sha256": sha256_file(dest)}


DOWNLOADERS: dict[str, Callable[..., dict[str, Any]]] = {
    "npd": download_npd,
    "plantdoc": download_plantdoc,
    "leafmap": download_leaf_map,
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--only", nargs="+", choices=sorted(DOWNLOADERS), default=sorted(DOWNLOADERS))
    p.add_argument("--allow-version-change", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prov_path = args.data_dir / "raw" / "PROVENANCE.json"
    prov: dict[str, Any] = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    for name in args.only:
        log.info("== %s", name)
        kwargs = {"allow_version_change": args.allow_version_change} if name == "npd" else {}
        record = DOWNLOADERS[name](args.data_dir, **kwargs)
        record["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prov[name] = record
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(prov, indent=2))
    log.info("provenance written to %s", prov_path)


if __name__ == "__main__":
    main()
