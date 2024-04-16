import os
import argparse
import torch
import utils
from transformers import AutoTokenizer
from model import Model
from dataset import PdyDataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", required=True)
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

    with torch.no_grad():
        correct_items, all_items = 0, 0
        for idx, data in enumerate(
            tqdm(
                DataLoader(
                    PdyDataset(config, args.file),
                    batch_size=16,
                    collate_fn=PdyDataset.collect,
                ),
                total=1000,
            )
        ):
            inputs_ids, attention_mask, token_type_ids, token_ids, type_ids = data
            pdy_hat, _, _ = model(inputs_ids, attention_mask, token_type_ids, token_ids)
            type_ids = type_ids.flatten()
            pdy_mask = type_ids != config["pdy_feats"]
            type_ids[~pdy_mask] = 0
            pdy_hat = torch.softmax(pdy_hat.reshape(type_ids.size(0), -1), dim=-1)
            pdy_hat = torch.argmax(pdy_hat, dim=-1)
            preds = (pdy_hat == type_ids) * pdy_mask
            correct_items += preds.sum().item()
            all_items += pdy_mask.sum().item()
            if idx > 50:
                break
        pdy_accu = correct_items / all_items

        print(f"{pdy_accu=}")


if __name__ == "__main__":
    main()
