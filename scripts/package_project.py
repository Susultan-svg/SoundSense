"""Package source, models, data and verified outputs without environment/cache files."""
import argparse
from pathlib import Path
import zipfile

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("../SoundSense.zip"))
    parser.add_argument("--omit-audio",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    files=[]
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve()==output:
            continue
        rel=path.relative_to(root)
        if set(rel.parts)&{".git",".venv","__pycache__",".pytest_cache"} or rel.parts[:3]==("data","features","cache"):
            continue
        if args.omit_audio and rel.parts[:3]==("data","raw","audio"):
            continue
        files.append(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path,Path("SoundSense")/path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failure: {bad}")
    print(f"Packaged and CRC-verified {len(files)} files ({output.stat().st_size/1e6:.1f} MB): {output}")

if __name__ == "__main__": main()
