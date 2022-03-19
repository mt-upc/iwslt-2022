from typing import Tuple
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import argparse
from examples.speech_to_text.data_utils import load_df_from_tsv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import json

tqdm.pandas(desc="progress")


class SourceDataset(Dataset):
    def __init__(self, data_root, asr_tsv_name, st_tsv_name, tokenizer, completed_ids):
        super().__init__()

        data_root = Path(data_root)

        asr_df = load_df_from_tsv(data_root / asr_tsv_name)
        st_df = load_df_from_tsv(data_root / st_tsv_name)
        assert (
            asr_df.id.tolist() == st_df.id.tolist()
        ), "Inconsistent datasets for ASR and ST"
        st_df["src_text"] = asr_df["tgt_text"].tolist()

        print("Tokenizing and sorting corpus ...")
        st_df["src_num_tokens"] = st_df.progress_apply(
            lambda x: len(tokenizer.tokenize(x["src_text"])), axis=1
        )
        st_df.sort_values(
            "src_num_tokens", inplace=True, ignore_index=True, ascending=False
        )

        self.data = [
            (id_, src_txt, tgt_txt)
            for id_, src_txt, tgt_txt in zip(
                st_df["id"].tolist(),
                st_df["src_text"].tolist(),
                st_df["tgt_text"].tolist(),
            )
            if id_ not in completed_ids
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Tuple[int, str, str]:
        return self.data[i]


def source_collate_fn(batch: list) -> Tuple[list[int], list[str], list[str]]:
    ids = [example[0] for example in batch]
    src_txt = [example[1] for example in batch]
    tgt_txt = [example[2] for example in batch]
    return (ids, src_txt, tgt_txt)


def prepare_corpus_for_kd(args):

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    print(f"Number of cuda devices: {n_gpu}")

    # specify main device and all devices (if gpu available)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")
    model = MBartForConditionalGeneration.from_pretrained(args.model_name).eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(
        args.model_name, src_lang="en_XX", tgt_lang=args.tgt_lang
    )

    if len(device_list) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_list, output_device=main_device
        )
    model.to(main_device)

    path_to_output = Path(args.path_to_output)
    if path_to_output.exists():
        completed_ids = [file.stem for file in path_to_output.glob(".pt")]
    else:
        completed_ids = []
        path_to_output.mkdir(parents=True, exist_ok=True)
        
    dataset = SourceDataset(
        args.data_root, args.asr_tsv_name, args.st_tsv_name, tokenizer, completed_ids
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size * n_gpu,
        collate_fn=source_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    with torch.no_grad():
        for sgm_ids, source_texts, target_texts in tqdm(iter(dataloader)):

            inputs = tokenizer(source_texts, return_tensors="pt", padding="longest")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(target_texts, return_tensors="pt", padding="longest")

            tgt_seq_lens = labels.attention_mask.sum(dim=1)

            inputs = inputs.to(main_device)
            labels = labels.to(main_device)

            logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels.input_ids,
            )["logits"]

            topk_outputs, topk_indices = torch.topk(logits, k=args.top_k, dim=-1)
            bs = len(topk_outputs)

            for i in range(bs):
                seq_len_i = tgt_seq_lens[i]
                topk_outputs_i = topk_outputs[i, 1:seq_len_i, :].detach().cpu()
                topk_indices_i = topk_indices[i, 1:seq_len_i, :].detach().cpu()
                
                torch.save(
                    {
                        "topk_indices": topk_indices_i,
                        "topk_outputs": topk_outputs_i,
                    }, args.path_to_output / f"{sgm_ids[i]}.pt"
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", type=str, required=True)
    parser.add_argument(
        "--asr-tsv-name",
        "-asr",
        required=True,
        type=str,
        help="path to the asr tsv with the english transcription under tgt_text",
    )
    parser.add_argument(
        "--st-tsv-name",
        "-st",
        required=True,
        type=str,
        help="path to the asr tsv with the english transcription under tgt_text",
    )
    parser.add_argument("--path-to-output", "-o", type=str, required=True)
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="facebook/mbart-large-50-one-to-many-mmt",
    )
    parser.add_argument("--tgt-lang", "-t", type=str, default="de_DE")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=8,
        help="keep the top-k most probable tokens",
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    args = parser.parse_args()

    prepare_corpus_for_kd(args)
