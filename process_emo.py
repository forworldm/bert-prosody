import argparse
import os
import regex
import json
import utils

from tqdm.auto import tqdm
from transformers import AutoTokenizer

emo_accepted = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "like",
    "none",
    "sad",
    "surprise",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--conf", required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.dir)
    assert os.path.isfile(args.conf)

    config = utils.get_config_from_file(args.conf)
    tker = AutoTokenizer.from_pretrained(
        os.path.realpath(config["pretrained_model"]),
        add_prefix_space=False,
        trust_remote_code=True,
    )
    data = []
    for file in sorted(os.listdir(args.dir)):
        file = os.path.join(args.dir, file)
        if not os.path.isfile(file) or not file.endswith(".txt"):
            continue
        with open(file, "r", encoding="utf-8") as file:
            for label in tqdm(file.readlines()):
                label = label.strip()
                if not label:
                    continue
                label, text = label.split(" ", 1)
                assert label in emo_accepted
                tks, ids = utils.tokenize_bert_format(tker, text)
                assert len(tks) > 2
                assert tks[0] == tker.cls_token and tks[-1] == tker.sep_token
                data.append([ids, emo_accepted.index(label)])

    valid_data, train_data = [], []
    ratio_num = 10
    ratio_den = 400
    for i in range(0, len(data), ratio_den):
        valid_data.extend(data[i : min(i + ratio_num, len(data))])
        if i + ratio_num < len(data):
            train_data.extend(data[i + ratio_num : min(i + ratio_den, len(data))])
    print(f"total {len(data)}")
    with open(os.path.join(args.dir, "valid.json"), "w", encoding="utf-8") as file:
        json.dump(valid_data, file)
    with open(os.path.join(args.dir, "train.json"), "w", encoding="utf-8") as file:
        json.dump(train_data, file)


if __name__ == "__main__":
    main()
