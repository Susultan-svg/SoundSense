import argparse
import json
from src.classification.predict import load_model, predict_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a WAV with a trusted local model")
    parser.add_argument("wav")
    parser.add_argument("--model",default="models/svm.joblib")
    args = parser.parse_args()
    print(json.dumps(predict_file(args.wav,load_model(args.model)),indent=2))

if __name__ == "__main__":
    main()
