import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path_to_en_scores = Path(sys.argv[1])
path_to_mult_scores = Path(sys.argv[2])

len_range = list(range(10, 31))
times_range = list(range(1, 6))

df = {f"{i}_{j}": pd.DataFrame(
        index=len_range,
        columns=["bleu", "ter"])
      for i in times_range for j in ["en", "mult"]}

for s, path_to_scores in zip(["en", "mult"], [path_to_en_scores, path_to_mult_scores]):
    for inference_times in times_range:
        
        results = {"bleu": .0, "ter": .0}
        for max_segment_length in len_range:
            
            name = f"{inference_times}_{s}"
            
            file_name = path_to_scores / f"{max_segment_length}_{inference_times}.json"
            try:
                with open(file_name) as f:
                    results = json.load(f)
                df[name].loc[max_segment_length] = list(results.values())
            except FileNotFoundError:
                print(f"{file_name} not found")
                df[name].loc[max_segment_length] = list(results.values())
                
        df[name] = df[name].astype(float)
        if df[name].bleu.sum() == 0:
            del df[name]
            continue
        else:
            df[name].loc[df[name].bleu == 0, :] = list(results.values())
        df[name] = df[name].astype(float)
        
        print(df[name])

        print("Inference times", name)
        print(f"Best BLEU of {df[name].bleu.max()} at max-segm-len of {df[name].bleu.idxmax()}")
        print(f"Best TER of {df[name].ter.min()} at max-segm-len of {df[name].ter.idxmin()}")
        print("_"*30)

for metric in ["bleu", "ter"]:
    plt.figure()
    plt.title(metric)
    for name in df.keys():
        plt.plot(len_range, df[name].loc[:, metric].tolist(), label=name)
    plt.legend()
    plt.savefig(path_to_en_scores.parent / f"{metric}.png", dpi=400)