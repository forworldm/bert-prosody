import argparse
import os
import regex
import json
import utils

from tqdm.auto import tqdm
from transformers import AutoTokenizer


symbol_regex = regex.compile("\\p{Punct}|\\p{Emoji}|\\p{Symbol}")
word_regex = regex.compile("(?:\\p{Alpha}|\\p{Digit}|')+")


def is_symbol(text):
    return regex.fullmatch(symbol_regex, text)


def is_word(text):
    if not text:
        return True
    if text[0].isalnum():
        return regex.fullmatch(word_regex, text)
    return False


def extract_pdy(label: str, tokens: list[str], unk_token: str):
    pdy_mask = 6
    pdy, ids = [pdy_mask], [0]  # [CLS]
    i, j = 1, 0
    next_pdy = 0
    while i < len(tokens) - 1:
        tk = tokens[i]
        n = 1
        while i + n < len(tokens) and tokens[i + n].startswith("##"):
            tk += tokens[i + n][2:]
            n += 1
            pdy.append(5 if tokens[i + n - 2][2:].isalnum() else 4)
            ids.append(i + n - 2)
        while j < len(label) and label[j].isspace():
            j += 1
        while j < len(label) and label[j] == "#":
            t = int(label[j + 1])
            assert t >= 1 and t <= 4
            next_pdy = min(t, 3)
            j += 2
        while j < len(label) and label[j].isspace():
            j += 1
        if tk == unk_token:
            l = 0
            while (
                j + l < len(label) and label[j + l].isascii() and label[j + l].isalpha()
            ):
                l += 1
            l = max(l, 1)
        else:
            l = len(tk)
            assert label[j : j + l] == tk
        assert j + l <= len(label)
        i += n
        j += l
        if is_word(tk) or (tk == unk_token and is_word(label[j - l : j])):
            pdy.append(next_pdy)
            next_pdy = 0
        else:
            assert is_symbol(label[j - l : j])
            pdy.append(4)
        ids.append(i - 1)  # the last piece of word
    while j < len(label) and label[j] == "#":
        j += 2
    assert i == len(tokens) - 1
    assert j == len(label)
    pdy.append(pdy_mask), ids.append(i)  # [SP]
    assert len(pdy) == len(ids) and len(pdy) == len(tokens) and max(pdy[1:-1]) < pdy_mask
    assert ids == list(range(0, len(ids)))
    return pdy, ids


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
                text = regex.sub("#\\d", "", label)
                tks, ids = utils.tokenize_bert_format(tker, text)
                assert tks[0] == tker.cls_token and tks[-1] == tker.sep_token
                tps, pos = extract_pdy(label, tks, tker.unk_token)
                data.append([ids, tps, pos])

    valid_data, train_data = [], []
    ratio_num = 10
    ratio_den = 200
    for i in range(0, len(data), ratio_den):
        valid_data.extend(data[i : min(i + ratio_num, len(data))])
        if i + ratio_num < len(data):
            train_data.extend(data[i + ratio_num : min(i + ratio_den, len(data))])
    with open(os.path.join(args.dir, "valid.json"), "w", encoding="utf-8") as file:
        json.dump(valid_data, file)
    with open(os.path.join(args.dir, "train.json"), "w", encoding="utf-8") as file:
        json.dump(train_data, file)


if __name__ == "__main__":
    main()
