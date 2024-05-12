from utils import process_docs_reduction, DESCRIPTION
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
from yaml import safe_load

DATASET_NAME = "ClimatePolicyRadar/national-climate-targets"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("tokenizer_name", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = process_docs_reduction(dataset)

    def map_fn(x):
        final_texts = []
        for description, question_text in zip(x["description"], x["question_text"]):
            final_texts.append(description + question_text)
        return tokenizer(final_texts, truncation=False)
    dataset = dataset.map(map_fn, batched=True)
    dataset.set_format(columns=["input_ids"])

    max_length = 0
    for example in dataset:
        max_length = max(max_length, len(example["input_ids"]))
    desc_len = len(tokenizer(DESCRIPTION, truncation=False)["input_ids"])

    print("\n ====== National Climate Targets Dataset ====== ")
    print(f"(Used tokenizer: {args.tokenizer_name})\n")
    print(f"Description length: {desc_len}")
    print(f"The longest question length: {max_length - desc_len}")
    print(f"Required total length: {max_length}")

    


