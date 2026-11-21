"""Official ESC-50 subset selection and fold/source integrity checks."""
from pathlib import Path
import hashlib
import json
import pandas as pd

CATEGORIES = {"clapping": "Hand clapping", "door_wood_knock": "Door knocking",
              "clock_alarm": "Clock alarm", "rain": "Rain", "dog": "Dog barking"}
METADATA_COLUMNS = ["filename", "category", "fold", "src_file"]

def validate_metadata(table: pd.DataFrame, strict: bool = True) -> None:
    required = set(METADATA_COLUMNS)
    if not required <= set(table) or table.empty or table[list(required)].isna().any().any():
        raise ValueError("Metadata requires nonempty filename, category, fold, src_file columns")
    if table.filename.duplicated().any() or not table.fold.isin(range(1, 6)).all():
        raise ValueError("Duplicate filename or invalid official fold")
    if any(Path(str(name)).name != str(name) or '/' in str(name) or '\\' in str(name) or not str(name).endswith('.wav') for name in table.filename):
        raise ValueError("Unsafe or invalid WAV filename")
    if table.groupby("src_file").fold.nunique().max() != 1:
        raise ValueError("Source recording crosses folds: leakage detected")
    if set(table.category) - set(CATEGORIES):
        raise ValueError("Unexpected sound category")
    if strict:
        counts = table.groupby(["category", "fold"]).size()
        if set(table.category) != set(CATEGORIES) or len(counts) != 25 or not (counts == 8).all():
            raise ValueError("Expected 5 classes x 5 official folds x 8 clips = 200 recordings")

def select_metadata(source: str | Path) -> pd.DataFrame:
    table = pd.read_csv(source)
    if "category" not in table or not set(CATEGORIES) <= set(table.category):
        raise ValueError("Official metadata does not contain all five exact category names")
    selected = table.loc[table.category.isin(CATEGORIES)].copy().sort_values("filename").reset_index(drop=True)
    validate_metadata(selected)
    return selected

def prepare_dataset(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    table = select_metadata(root / "meta" / "esc50.csv")
    missing = [name for name in table.filename if not (root / "audio" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files missing; first: {missing[0]}")
    table.to_csv(root / "selected_metadata.csv", index=False)
    return table

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dataset_fingerprint(table: pd.DataFrame) -> str:
    return hashlib.sha256(table[METADATA_COLUMNS].sort_values("filename").to_csv(index=False).encode()).hexdigest()
