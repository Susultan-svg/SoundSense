"""Train on clean folds; add seeded power-scaled noise to held-out recordings only."""
import hashlib
import numpy as np
import pandas as pd
from src.preprocessing.audio_loader import load_wav, preprocess_audio
from src.frequency_analysis.spectral_analysis import add_white_noise, measured_snr
from src.features.feature_extractor import extract_features
from .common import Experiment, arguments

def main() -> None:
    e=Experiment(arguments("noise_robustness")); clean=e.features(e.config)
    e.run("clean",e.config,{"snr_db":"clean"},train_table=clean)
    snr_records=[]
    for snr in (30,20,10,5,0):
        rows=[]
        for row in e.table.itertuples():
            x,sr=load_wav(e.args.root/"audio"/row.filename)
            seed=int.from_bytes(hashlib.sha256(f"{e.config.seed}:{row.filename}".encode()).digest()[:8],"little")
            noisy=add_white_noise(x,snr,np.random.default_rng(seed))
            snr_records.append({"filename":row.filename,"fold":row.fold,"requested_snr_db":snr,"measured_snr_db":measured_snr(x,noisy)})
            processed,target=preprocess_audio(noisy,sr,e.config.preprocessing)
            rows.append({"filename":row.filename,"category":row.category,"fold":row.fold,"src_file":row.src_file,
                         **extract_features(processed,target,e.config.features)})
        noisy_table=pd.DataFrame(rows)
        e.run(f"snr{snr}dB",e.config,{"snr_db":snr},train_table=clean,test_table=noisy_table)
        noisy_table.to_csv(e.output/f"snr{snr}dB"/"noisy_features.csv",index=False)
    pd.DataFrame(snr_records).to_csv(e.output/"snr_verification.csv",index=False)
    e.finish()

if __name__ == "__main__": main()
