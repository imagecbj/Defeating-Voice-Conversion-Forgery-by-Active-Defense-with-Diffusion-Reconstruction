from pathlib import Path
import numpy as np
import yaml
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from hyperparameter import hp

voiceEncoder = VoiceEncoder()

def compute_sim(target, vc):
    threshold_path = hp.vctk_eer
    info = yaml.safe_load(Path(threshold_path).open())
    emb_a = voiceEncoder.embed_utterance(preprocess_wav(target))
    emb_b = voiceEncoder.embed_utterance(preprocess_wav(vc))
    cosine_similarity = (
        np.inner(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)
    )
    return True if cosine_similarity > info["Threshold"] else False, cosine_similarity
    # target  vc

def evaluate_sim(dataset1_root, dataset2_root):
    n_accept = 0
    dataset1_root = Path(dataset1_root)
    dataset2_root = Path(dataset2_root)
    
    # 1
    dataset1_files = sorted(list(dataset1_root.glob("**/*.wav")))
    total = len(dataset1_files)
    
    if total == 0:
        print("1.wav")
        return 0.0

    for file1 in tqdm(dataset1_files):
        filename1 = file1.stem  # 
        # 2filename1
        matching_files = list(dataset2_root.glob(f"**/*{filename1}*.wav"))
        if matching_files:
            file2 = matching_files[0]  # 
            sim_record = compute_sim(file1, file2)
            # 
            print(f"\n  1: {file1}\n  2: {file2}\n  : {sim_record[1]:.4f}")
            if sim_record[0]:
                n_accept += 1
        else:
            print(f"2 {filename1} ")
    acceptance_ratio = n_accept / total
    print(f"\n{acceptance_ratio * 100:.2f}%")
    return acceptance_ratio
    # 12

if __name__ == "__main__":
    dataset1_root =""
    dataset2_root = ""
    evaluate_sim(dataset1_root, dataset2_root)