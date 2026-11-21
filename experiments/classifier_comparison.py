from src.classification.train import ModelSpec
from .common import Experiment, arguments

def main() -> None:
    args=arguments("classifier_comparison")
    if args.tune:
        raise ValueError("Representative fixed K/kernel comparison must not tune away the nominated settings; use train_models --tune for nested selection")
    e=Experiment(args)
    for k in (1,3,5,7,9):
        e.run(f"knn_k{k}",e.config,{"k":k},[ModelSpec("knn",k=k)])
    for kernel in ("linear","rbf","poly"):
        e.run(f"svm_{kernel}",e.config,{"kernel":kernel,"C":1.,"gamma":"scale","degree":3},[ModelSpec("svm",kernel=kernel)])
    e.finish()

if __name__ == "__main__": main()
