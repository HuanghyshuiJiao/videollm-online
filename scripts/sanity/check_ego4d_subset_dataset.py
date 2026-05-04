from dataclasses import asdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import build_concat_train_dataset, build_eval_dataset_dict
from models import parse_args
from models.tokenization_live import build_live_tokenizer_and_update_config
from models.live_llama.configuration_live_llama import LiveLlamaConfig


def main():
    args = parse_args()
    config = LiveLlamaConfig.from_pretrained(args.llm_pretrained, **asdict(args))
    tokenizer = build_live_tokenizer_and_update_config(args.llm_pretrained, config)
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_datasets = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    print(f"train_len={len(train_dataset)}")
    for name, dataset in eval_datasets.items():
        print(f"{name}_len={len(dataset)}")
    sample = train_dataset[0]
    print(f"sample_text_chars={len(sample[0])}")
    print(f"sample_frames_shape={tuple(sample[1].shape)}")
    print(f"sample_learn_ranges={len(sample[2])}")


if __name__ == "__main__":
    main()
