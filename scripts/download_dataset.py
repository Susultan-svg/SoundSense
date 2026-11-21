"""Download only the 200 requested WAVs, pinned to a resolved upstream commit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
from urllib.request import urlopen, Request
from src.preprocessing.dataset import select_metadata, prepare_dataset, sha256
from src.preprocessing.audio_loader import load_wav

def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "SoundSense-course-project"}), timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Unreachable")

def download(root: Path, revision: str = "master", workers: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if revision == "master" and previous.get("commit"):
        commit = previous["commit"]
    elif len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        commit = revision
    else:
        commit = json.loads(fetch(f"https://api.github.com/repos/karolpiczak/ESC-50/commits/{revision}"))["sha"]
    base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{commit}"
    (root / "meta").mkdir(exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    for relative in ("meta/esc50.csv", "LICENSE"):
        (root / relative).write_bytes(fetch(f"{base}/{relative}"))
    table = select_metadata(root / "meta/esc50.csv")
    hashes = previous.get("sha256", {}) if previous.get("commit") == commit else {}
    def one(name: str) -> tuple[str, str]:
        path = root / "audio" / name
        if not path.exists() or name not in hashes or sha256(path) != hashes[name]:
            payload = fetch(f"{base}/audio/{name}")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(payload)
            x, sr = load_wav(temporary)
            if len(x) != 5 * sr:
                raise ValueError(f"Unexpected duration: {name}")
            temporary.replace(path)
        return name, sha256(path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, digest) in enumerate(pool.map(one, table.filename), 1):
            hashes[name] = digest
            manifest_path.write_text(json.dumps({"commit": commit, "source": base, "sha256": hashes,
                "metadata_sha256": sha256(root / "meta/esc50.csv")}, indent=2), encoding="utf-8")
            if i % 20 == 0:
                print(f"Verified {i}/{len(table)} WAV files", flush=True)
    prepare_dataset(root)
    print(f"Prepared {len(table)} files at revision {commit}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download(args.root, args.revision, args.workers)

if __name__ == "__main__":
    main()
