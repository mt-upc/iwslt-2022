import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

import argparse

import datasets

from knowledge_distillation.source_dataset import SourceDataset


def mbart_generation(args):

    device = torch.device("cuda:0")

    model = MBartForConditionalGeneration.from_pretrained(args.model_name).eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(
        args.model_name, src_lang="en_XX", tgt_lang=args.tgt_lang
    )
    model.to(device)

    dataset = SourceDataset(
        args.data_root, args.asr_tsv_name, args.st_tsv_name, None, []
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collater,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    all_translations, all_references = [], []
    with torch.no_grad():
        for _, source_texts, target_texts in tqdm(iter(dataloader)):

            inputs = tokenizer(source_texts, return_tensors="pt", padding="longest")
            inputs = inputs.to(device)

            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang]
            )
            translations = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            all_translations.extend(translations)
            all_references.extend(target_texts)

    sacrebleu = datasets.load_metric("sacrebleu")

    all_references = [[ref] for ref in all_references]

    if args.tgt_lang == "de_DE":
        results = sacrebleu.compute(
            predictions=all_translations, references=all_references
        )
    elif args.tgt_lang == "ja_XX":
        results = sacrebleu.compute(
            predictions=all_translations, references=all_references, tokenize="ja-mecab"
        )
    else:
        results = sacrebleu.compute(
            predictions=all_translations, references=all_references, tokenize="zh"
        )

    print(results["score"])


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
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="facebook/mbart-large-50-one-to-many-mmt",
    )
    parser.add_argument("--tgt-lang", "-t", type=str, default="de_DE")
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    args = parser.parse_args()

    mbart_generation(args)
