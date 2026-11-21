"""Execute all six standard experiments in separate reproducible CLI processes."""
import subprocess
import sys

def main() -> None:
    for name in ("window_length","window_comparison","feature_comparison","classifier_comparison","noise_robustness","filter_comparison"):
        subprocess.run([sys.executable,"-m",f"experiments.{name}",*sys.argv[1:]],check=True)

if __name__ == "__main__": main()
