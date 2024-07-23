import torch
import json
import os
import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


pad_token_id = None

class PdyDataset(Dataset):
    def __init__(self, config, file):
        super().__init__()
        self.max_len = config["max_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.realpath(config["pretrained_model"]),
            add_prefix_space=False,
            trust_remote_code=True,
        )
        global pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        self.items = []
        self.lengths = []
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        random.seed(1234)
        random.shuffle(data)
        for tks, tps, ids in data:
            if len(tks) > self.max_len:
                for i in range(len(ids) - 1, -1, -1):
                    if ids[i] + 1 < self.max_len and tps[i] != config["pdy_feats"] - 1:
                        break
                if i < 0:
                    continue
                tks = tks[0 : ids[i] + 1] + tks[-1:]
                tps = tps[0 : i + 1] + tps[-1:]
                ids = ids[0 : i + 1] + [len(tks) - 1]
            self.items.append((tks, tps, ids))
            self.lengths.append(len(tks))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def collect(data):
        tk_len = max(len(it[0]) for it in data)
        tp_len = max(len(it[1]) for it in data)
        inputs_ids, attention_mask, token_type_ids = [], [], []
        token_ids, type_ids = [], []
        for tks, tps, ids in data:
            pad_id = pad_token_id  # [PAD]
            inputs_ids_it = tks + [pad_id] * (tk_len - len(tks))
            attention_mask_it = [1] * len(tks) + [0] * (tk_len - len(tks))
            token_type_ids_it = [0] * tk_len

            sp_id = ids[-1]  # [SP]
            sp_tp = tps[-1]
            token_ids_it = ids + list(range(sp_id + 1, sp_id + 1 + tp_len - len(ids)))
            type_ids_it = tps + [sp_tp] * (tp_len - len(tps))

            inputs_ids.append(inputs_ids_it)
            attention_mask.append(attention_mask_it)
            token_type_ids.append(token_type_ids_it)
            token_ids.append(token_ids_it)
            type_ids.append(type_ids_it)
        return (
            torch.LongTensor(inputs_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(token_ids),
            torch.LongTensor(type_ids),
        )


class EmoDataset(Dataset):
    def __init__(self, config, file):
        super().__init__()
        self.max_len = config["max_len"]
        self.items = []
        self.lengths = []
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        random.seed(1234)
        random.shuffle(data)
        for tks, tp in data:
            if len(tks) > self.max_len:
                tks = tks[0 : self.max_len - 1] + tks[-1:]
            self.items.append((tks, tp))
            self.lengths.append(len(tks))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def collect(data):
        tk_len = max(len(it[0]) for it in data)
        inputs_ids, attention_mask, token_type_ids = [], [], []
        token_ids, type_ids = [], []
        for tks, tp in data:
            pad_id = pad_token_id  # [PAD]
            inputs_ids_it = tks + [pad_id] * (tk_len - len(tks))
            attention_mask_it = [1] * len(tks) + [0] * (tk_len - len(tks))
            token_type_ids_it = [0] * tk_len
            inputs_ids.append(inputs_ids_it)
            attention_mask.append(attention_mask_it)
            token_type_ids.append(token_type_ids_it)
            token_ids.append(len(tks) - 1)
            type_ids.append(tp)
        return (
            torch.LongTensor(inputs_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(token_ids),
            torch.LongTensor(type_ids),
        )


class SimDataset(Dataset):
    def __init__(self, config, file):
        super().__init__()
        self.max_len = config["max_len"]
        self.items = []
        self.lengths = []
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        random.seed(4321)
        random.shuffle(data)
        for tks, tp in data:
            nxt_tks, nxt_tp = random.choice(data)
            l = len(tks) + len(nxt_tks)
            if len(tks) + len(nxt_tks) > self.max_len:
                if len(tks) > self.max_len // 2 and len(nxt_tks) > self.max_len // 2:
                    l1 = len(tks) * self.max_len // l
                    l2 = len(nxt_tks) * self.max_len // l
                    tks = tks[0:l1] + tks[-1:]
                    nxt_tks = nxt_tks[1 : l2 - 1] + nxt_tks[-1:]
                elif len(tks) > len(nxt_tks):
                    l1 = self.max_len - len(nxt_tks)
                    tks = tks[0 : l1 - 1] + tks[-1:]
                else:
                    l2 = self.max_len - len(tks)
                    nxt_tks = nxt_tks[1 : l2 - 1] + nxt_tks[-1:]
            self.items.append((tks, nxt_tks, int(tp == nxt_tp)))
            self.lengths.append(len(tks) + len(nxt_tks))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def collect(data):
        tk_len = max(len(it[0]) + len(it[1]) for it in data)
        inputs_ids, attention_mask, token_type_ids = [], [], []
        type_ids = []
        for tks, nxt_tks, tp in data:
            l = len(tks) + len(nxt_tks)
            pad_id = pad_token_id  # [PAD]
            inputs_ids_it = tks + nxt_tks + [pad_id] * (tk_len - l)
            attention_mask_it = [1] * l + [0] * (tk_len - l)
            token_type_ids_it = [0] * len(tks) + [1] * (tk_len - len(tks))
            inputs_ids.append(inputs_ids_it)
            attention_mask.append(attention_mask_it)
            token_type_ids.append(token_type_ids_it)
            type_ids.append(tp)
        return (
            torch.LongTensor(inputs_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(type_ids),
        )
