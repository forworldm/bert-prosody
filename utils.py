import json
import os
import sys
import re
import glob
import torch
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def get_config_from_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_checkpoint_path(dir_path, regex):
    files = glob.glob(os.path.join(dir_path, regex))
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if not files:
        return None
    x = files[-1]
    print(x)
    return x


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint_dict["epoch"]
    learning_rate = checkpoint_dict["learning_rate"]

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    saved_state_dict = checkpoint_dict["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info(f"loaded checkpoint {checkpoint_path}, {epoch=}")
    iteration_step = 0
    step_match = re.search("\\d+", os.path.basename(checkpoint_path))
    if step_match:
        iteration_step = int(step_match[0])
    return learning_rate, epoch, iteration_step


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path):
    logger.info(f"Saving model and optimizer state at {epoch=} to {checkpoint_path}")
    state_dict = model.state_dict()
    checkpoint_dict = {
        "model": state_dict,
        "epoch": epoch,
        "learning_rate": learning_rate,
    }
    if optimizer is not None:
        checkpoint_dict["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint_dict, checkpoint_path)


import regex

_word_regex = regex.compile("\\s*(?:[A-Za-z]+(?:'(?:s|t|re|ve|m|ll|d)(?=[^A-Za-z]))?|\\d+|\\X)")

def tokenize_bert_format(tokenizer, text):
    words = regex.findall(_word_regex, text)
    tokens, ids = [tokenizer.cls_token], [tokenizer.cls_token_id]
    for w in words:
        t = tokenizer(w, add_special_tokens=False, return_offsets_mapping=True)
        i = len(tokens)
        for s in t["offset_mapping"]:
            s = w[s[0] : s[1]].strip()
            tokens.append(s if i == len(tokens) else "##" + s)
        ids.extend(t["input_ids"])
    tokens.append(tokenizer.sep_token)
    ids.append(tokenizer.sep_token_id)
    assert len(tokens) == len(ids)
    return tokens, ids
