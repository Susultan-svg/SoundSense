"""Validate an existing official ESC-50 checkout and optionally write processed WAVs."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from src.config import load_config
from src.preprocessing.dataset import prepare_dataset
from src.preprocessing.audio_loader import preprocess_file

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    table = prepare_dataset(args.root)
    if args.processed:
        config = load_config(args.config)
        args.processed.mkdir(parents=True, exist_ok=True)
        for name in table.filename:
            x, sr = preprocess_file(args.root / "audio" / name, config.preprocessing)
            write(args.processed / name, sr, x.astype(np.float32))
        table.to_csv(args.processed / "metadata.csv", index=False)
        import json
        (args.processed / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    print(table.groupby(["category", "fold"]).size().unstack())

if __name__ == "__main__":
    main()
