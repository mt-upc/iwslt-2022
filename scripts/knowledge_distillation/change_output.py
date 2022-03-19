from posixpath import split
import torch
from pathlib import Path
import json
from tqdm import tqdm

KD_ROOT="/home/usuaris/veussd/DATABASES/speech_translation/IWSLT22/knowledge_distillation"

for lang_pair_dir in Path(KD_ROOT).iterdir():
    for json_file in lang_pair_dir.glob("*.json"):
        
        with open(json_file, "r") as f:
            kd_content = [json.loads(example) for example in f]
            
        split_dir = json_file.parent / json_file.stem
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(lang_pair_dir.name, split_dir.name)
        for example in tqdm(kd_content):
            torch.save(
                {
                    "topk_indices": torch.tensor(example["topk_indices"]),
                    "topk_outputs": torch.tensor(example["topk_outputs"])
                }, split_dir / f"{example['id']}.pt"
            )