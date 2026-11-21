from dataclasses import replace
from .common import Experiment, arguments

def main() -> None:
    e=Experiment(arguments("feature_comparison"))
    for name,groups in [("spectral",("spectral",)),("mfcc",("mfcc",)),("temporal",("temporal",)),("combined",("spectral","temporal","mfcc"))]:
        e.run(name,replace(e.config,features=replace(e.config.features,groups=groups)))
    e.finish()
    best=e.comparison().sort_values("f1_macro_mean",ascending=False).iloc[0]
    (e.output/"interpretation.txt").write_text(f"Largest observed macro F1: {best['variant']} / {best['algorithm']} = {best['f1_macro_mean']:.6f}. Exploratory comparison on shared outer folds; no independent post-selection performance claim.\n")

if __name__ == "__main__": main()
