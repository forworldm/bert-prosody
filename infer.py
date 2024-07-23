import os
import argparse
import torch
import utils
from transformers import AutoTokenizer
from model import Model
from process_emo import emo_accepted


def extract_pdy_map(label: str, tokens: list[str], unk_token: str):
    ids, pos = [0], []
    i, j = 1, 0
    while i < len(tokens) - 1:
        tk = tokens[i]
        n = 1
        while i + n < len(tokens) and tokens[i + n].startswith("##"):
            tk += tokens[i + n][2:]
            n += 1
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
        ids.append(i - 1)
        pos.append(j)
    assert i == len(tokens) - 1
    assert j >= len(label)
    ids.append(i)
    return ids, pos


def eval_pdy(tokenizer, model, text):
    tks, ids = utils.tokenize_bert_format(tokenizer, text)
    tps, pos = extract_pdy_map(text, tks, tokenizer.unk_token)
    input_ids = torch.LongTensor(ids).unsqueeze(0)
    attention_mask = torch.ones(input_ids.size(-1), dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.zeros(input_ids.size(-1), dtype=torch.long).unsqueeze(0)
    tps = torch.LongTensor(tps).unsqueeze(0)
    pdy, _, _ = model(input_ids, attention_mask, token_type_ids, tps)
    pdy = torch.softmax(pdy, dim=-1)
    pdy = torch.argmax(pdy, dim=-1).tolist()[0]
    text = list(text)
    for i, p in zip(reversed(pos), reversed(pdy[1:-1])):
        if p != 0:
            text.insert(i-1, f"#{p}")
    text = "".join(text)
    print(text)


def eval_emo(tokenizer, model, text):
    res = tokenizer(text)
    input_ids = torch.LongTensor(res["input_ids"]).unsqueeze(0)
    attention_mask = torch.ones(input_ids.size(-1), dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.zeros(input_ids.size(-1), dtype=torch.long).unsqueeze(0)
    _, emo, _ = model(input_ids, attention_mask, token_type_ids, None, nsp=False)
    emo = torch.softmax(emo, dim=-1)
    print({emo_accepted[i]: v for i, v in enumerate(emo.tolist()[0])})
    emo = torch.argmax(emo, dim=-1)
    emo = emo_accepted[emo.tolist()[0]]
    print(f"{emo=}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = utils.get_config_from_file(args.conf)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.realpath(config["pretrained_model"]),
        add_prefix_space=False,
        trust_remote_code=True,
    )
    model = Model(config, from_pretrained=False)
    utils.load_checkpoint(args.model, model)
    model.eval()

    while True:
        try:
            text = input("type text...")
        except EOFError:
            break
        if not text:
            continue
        t, text = text.split(" ", maxsplit=1)
        with torch.no_grad():
            if t == "emo":
                eval_emo(tokenizer, model, text)
            elif t == "pdy":
                eval_pdy(tokenizer, model, text)


if __name__ == "__main__":
    main()
