import json
from pathlib import Path
import torch
from tqdm import tqdm
from data.utils import get_melgan_mel, load_json, gender_info
from hyperparameter import hp

def vctk_processing_wav(data_dir, mel_dir):
    data_dir = Path(data_dir)
    mel_dir = Path(mel_dir)
    mel_dir.mkdir(exist_ok=True)

    spk_dirs = data_dir.iterdir()
    for spk in tqdm(spk_dirs):
        if spk.name == ".DS_Store":
            continue

        wav_files = list(spk.glob("*.wav"))
        (mel_dir / spk.name).mkdir(exist_ok=True)
        for wav in wav_files:
            print(f"Processing WAV: {wav.stem}")
            mel = get_melgan_mel(wav)
            # 
            torch.save(mel.cpu(), mel_dir / spk.name / f"{wav.stem}.pt")

    print("WAV processing complete!")

def metadata(mel_dir, audio_dir, std_dir):
    mel_dir = Path(mel_dir)
    audio_dir = Path(audio_dir)
    std_dir = Path(std_dir)
    gender_dict = load_json(hp.gender_info)
    print(gender_dict)
    meta_data = dict()
    meta_data["train"] = []
    meta_data["valid"] = []
    meta_data["test"] = []

    for spk in mel_dir.iterdir():
        if spk.is_dir():
            mel_files = list(spk.glob("*.pt"))
            for mel_file in mel_files:
                mel_path = mel_file
                # 
                audio_path = audio_dir / spk.name / f"{mel_file.stem}.wav"
                std_path = std_dir / spk.name / f"{mel_file.stem}_std.pt"  # 

                if spk.name in gender_dict["female"]:
                    gender = 0
                else:
                    gender = 1

                data_entry = (str(mel_path), gender, str(audio_path), str(std_path))

                if spk.name in hp.test_set:
                    meta_data["test"].append(data_entry)
                elif spk.name in hp.valid_set:
                    meta_data["valid"].append(data_entry)
                else:
                    meta_data["train"].append(data_entry)

    with open(hp.metadata, "w") as j:
        json.dump(meta_data, j)

    print("Metadata creation complete!")

# 
mel_directory = ""
audio_directory = ""
std_directory = ""  # 
metadata(mel_directory, audio_directory, std_directory)