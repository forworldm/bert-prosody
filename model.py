import torch
import os
from torch import nn
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from peft import get_peft_model, LoraConfig


def mean_pooling(token_embeddings, attention_mask):
    token_mask = attention_mask.unsqueeze(-1).float()
    return torch.sum((token_embeddings * token_mask).float(), 1) / torch.clamp(token_mask.sum(1), min=1e-9)


class Model(nn.Module):
    def __init__(self, config, from_pretrained=True, lora=False):
        super().__init__()
        self.config = config
        self.bert = None
        self.transform = None
        if from_pretrained:
            if self.bert is None:
                self.bert = AutoModel.from_pretrained(
                    os.path.realpath(config["pretrained_model"]),
                    trust_remote_code=True,
                    add_pooling_layer=False,
                )
        else:
            bert_config = AutoConfig.from_pretrained(
                os.path.realpath(config["pretrained_model"]),
                trust_remote_code=True,
            )
            self.bert = AutoModel.from_config(
                bert_config,
                trust_remote_code=True,
                add_pooling_layer=False,
            )
        if lora:
            self.bert = get_peft_model(
                self.bert,
                LoraConfig(
                    target_modules="all-linear",
                    r=config["lora_rank"],
                    lora_alpha=config["lora_alpha"],
                ),
            )
        if self.transform is None:
            self.transform = BertPredictionHeadTransform(self.bert.config)
        hidden_size = self.bert.config.hidden_size
        self.pdy_cls = nn.Linear(hidden_size, config["pdy_feats"])
        self.emo_cls = nn.Linear(hidden_size, config["emo_feats"])
        self.sim_cls = nn.Linear(hidden_size, 2)

    def forward(self, inputs_ids, attention_mask, token_type_ids, token_ids, nsp=False):
        out_seq = self.bert(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )["last_hidden_state"]
        out_seq = self.transform(out_seq)
        pdy, emo, sim = None, None, None
        if (token_ids is None or token_ids.dim() == 1) and self.config["pooler"] == "mean":
            out_seq = mean_pooling(out_seq, attention_mask)
            if nsp:
                sim = self.sim_cls(out_seq)
            else:
                emo = self.emo_cls(out_seq)
        elif token_ids is None:
            if nsp:
                sim = self.sim_cls(out_seq[:, 0, :])
            else:
                emo = self.emo_cls(out_seq[:, 0, :])
        elif token_ids.dim() == 1:
            assert not nsp
            token_ids = token_ids.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, out_seq.size(-1)))
            out_seq = out_seq.gather(1, token_ids)
            emo = self.emo_cls(out_seq[:, 0, :])
        else:
            assert token_ids.dim() == 2
            token_ids = token_ids.unsqueeze(-1).expand((-1, -1, out_seq.size(-1)))
            out_seq = out_seq.gather(1, token_ids)
            pdy = self.pdy_cls(out_seq)
        return (pdy, emo, sim)
