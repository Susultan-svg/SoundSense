import argparse
from pathlib import Path
from src.config import load_config
from src.preprocessing.dataset import prepare_dataset
from src.features.feature_extractor import extract_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fold-preserving spectral features")
    parser.add_argument("--root",type=Path,default=Path("data/raw"))
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--output",type=Path,default=Path("data/features/features.csv"))
    args = parser.parse_args()
    result = extract_dataset(prepare_dataset(args.root),args.root/"audio",load_config(args.config),args.output)
    print(f"Saved {len(result)} recordings, {len(result.columns)-4} features to {args.output}")

if __name__ == "__main__":
    main()
