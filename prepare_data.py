from data import metadata, vctk_processing_wav
from data.utils import vctk_48_2_22
from hyperparameter import hp

def main():
    # 
    std_dir = "/home/lab/workspace/works/thy/data/std"  # 
    metadata(hp.vctk_mels, std_dir)  # 

if __name__ == "__main__":
    main()