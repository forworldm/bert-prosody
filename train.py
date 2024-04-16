import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import torch
import argparse
import utils
import platform
import os

from os import path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from dataset import PdyDataset, EmoDataset, SimDataset
from dist_dataset import DistributedBucketSampler
from transformers import get_linear_schedule_with_warmup
from model import Model


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
amp_type = torch.bfloat16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", required=True)
    parser.add_argument("--pdy-dir", required=True)
    parser.add_argument("--emo-dir", required=True)
    parser.add_argument("--tune-layers", type=int, default=0)
    parser.add_argument("--no-optim", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    print(args)
    config = utils.get_config_from_file(args.conf)
    config["model_dir"] = args.model

    dist.init_process_group(
        backend="gloo" if platform.system().lower() == "windows" else "nccl",
        init_method="env://",
        world_size=1,
        rank=0,
    )
    torch.manual_seed(config["seed"])
    logger = utils.get_logger(args.model)
    writer = SummaryWriter(log_dir=args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"train on {device=}")
    model = Model(config, lora=(args.tune_layers == -3))
    model = model.to(device)
    model.train()

    if args.tune_layers >= 0:
        for p in model.bert.parameters():
            p.requires_grad = False
        bert_layers = model.bert.encoder.layer
        for i, m in enumerate(bert_layers):
            if i >= len(bert_layers) - args.tune_layers:
                for p in m.parameters():
                    p.requires_grad = True
    elif args.tune_layers == -2:
        for p in model.bert.embeddings.position_embeddings.parameters():
            p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [
                p
                for n, p in model.bert.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
            "lr": config["bert_lr"],
        },
        {
            "params": [
                p
                for n, p in model.bert.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": config["bert_lr"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not "bert" in n and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not "bert" in n and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_params, lr=config["lr"])

    start_epoch = 1
    global_step = 0
    ckpt_file = utils.latest_checkpoint_path(args.model, "pdy-*.pth")
    if ckpt_file:
        _, start_epoch, global_step = utils.load_checkpoint(
            ckpt_file, model, None if args.no_optim else optimizer
        )

    num_works = 0 if platform.system().lower() == "windows" else 4
    pdy_train_data = PdyDataset(config, path.join(args.pdy_dir, "train.json"))
    pdy_train_sampler = DistributedBucketSampler(
        pdy_train_data,
        batch_size=config["batch_size"],
        boundaries=[4, 224, 320, 496],
        shuffle=True,
    )
    pdy_train_loader = DataLoader(
        pdy_train_data,
        shuffle=False,
        num_workers=num_works,
        batch_sampler=pdy_train_sampler,
        collate_fn=PdyDataset.collect,
    )
    pdy_valid_loader = DataLoader(
        PdyDataset(config, path.join(args.pdy_dir, "valid.json")),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=num_works,
        collate_fn=PdyDataset.collect,
    )

    emo_train_data = EmoDataset(config, path.join(args.emo_dir, "train.json"))
    emo_train_sampler = DistributedBucketSampler(
        emo_train_data,
        batch_size=config["batch_size"],
        boundaries=[4, 224, 320, 496],
        shuffle=True,
    )
    emo_train_loader = DataLoader(
        emo_train_data,
        shuffle=False,
        num_workers=num_works,
        batch_sampler=emo_train_sampler,
        collate_fn=EmoDataset.collect,
    )
    emo_valid_loader = DataLoader(
        EmoDataset(config, path.join(args.emo_dir, "valid.json")),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=num_works,
        collate_fn=EmoDataset.collect,
    )

    sim_train_data = SimDataset(config, path.join(args.emo_dir, "train.json"))
    sim_train_sampler = DistributedBucketSampler(
        sim_train_data,
        batch_size=config["batch_size"],
        boundaries=[4, 224, 320, 496],
        shuffle=True,
    )
    sim_train_loader = DataLoader(
        sim_train_data,
        shuffle=False,
        num_workers=num_works,
        batch_sampler=sim_train_sampler,
        collate_fn=SimDataset.collect,
    )
    sim_valid_loader = DataLoader(
        SimDataset(config, path.join(args.emo_dir, "valid.json")),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=num_works,
        collate_fn=SimDataset.collect,
    )

    print(
        f"train-dataset: {len(pdy_train_loader)} {len(emo_train_loader)} {len(sim_train_loader)}"
    )
    print(
        f"eval-dataset: {len(pdy_valid_loader)} {len(emo_valid_loader)} {len(sim_valid_loader)}"
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(
            0 if args.no_warmup else config["warmup_epochs"] * len(emo_train_loader)
        ),
        num_training_steps=config["num_epochs"] * len(emo_train_loader),
        last_epoch=(-1 if args.no_optim else global_step - 1),
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=config["fp16_run"] and amp_type == torch.float16)
        if device == "cuda"
        else None
    )

    train(
        config,
        logger,
        start_epoch,
        global_step,
        device,
        model,
        optimizer,
        scaler,
        scheduler,
        (
            pdy_train_loader,
            pdy_valid_loader,
            emo_train_loader,
            emo_valid_loader,
            sim_train_loader,
            sim_valid_loader,
        ),
        writer,
    )


def train(
    config,
    logger,
    epoch,
    global_step,
    device,
    model: Model,
    optimizer: torch.optim.AdamW,
    scaler: torch.cuda.amp.GradScaler,
    scheduler,
    loaders,
    writer: SummaryWriter,
):
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def backward(loss):
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

    pdy_train, pdy_valid, emo_train, emo_valid, sim_train, sim_valid = loaders
    pdy_train.batch_sampler.set_epoch(epoch)
    emo_train.batch_sampler.set_epoch(epoch)
    sim_train.batch_sampler.set_epoch(epoch)
    pdy_iter = iter(tqdm(pdy_train))
    emo_iter = iter(emo_train)
    sim_iter = iter(sim_train)
    while epoch < config["num_epochs"] + 1:
        try:
            data = next(pdy_iter)
        except StopIteration:
            print("new pdy...")
            pdy_train.batch_sampler.set_epoch(epoch)
            pdy_iter = iter(tqdm(pdy_train))
            data = next(pdy_iter)
        inputs_ids, attention_mask, token_type_ids, token_ids, type_ids = (
            move_to_device(data, device)
        )
        with torch.autocast(
            device_type=device, enabled=config["fp16_run"], dtype=amp_type
        ):
            pdy_hat, _, _ = model(inputs_ids, attention_mask, token_type_ids, token_ids)
        type_ids = type_ids.flatten()
        pdy_mask = type_ids != config["pdy_feats"]
        type_ids[~pdy_mask] = 0
        pdy_loss = (
            loss_func(pdy_hat.reshape(type_ids.size(0), -1).float(), type_ids) * pdy_mask
        ).sum() / pdy_mask.sum()
        optimizer.zero_grad()
        backward(pdy_loss * config["c_pdy"])

        try:
            data = next(emo_iter)
        except StopIteration:
            print("new emo...")
            epoch += 1
            emo_train.batch_sampler.set_epoch(epoch)
            emo_iter = iter(emo_train)
            data = next(emo_iter)
        inputs_ids, attention_mask, token_type_ids, token_ids, type_ids = move_to_device(
            data, device
        )
        with torch.autocast(
            device_type=device, enabled=config["fp16_run"], dtype=amp_type
        ):
            _, emo_hat, _ = model(
                inputs_ids, attention_mask, token_type_ids, token_ids, nsp=False
            )
        emo_loss = loss_func(emo_hat.float(), type_ids).mean()
        backward(emo_loss * config["c_emo"])

        try:
            data = next(sim_iter)
        except StopIteration:
            print("new sim...")
            sim_train.batch_sampler.set_epoch(epoch)
            sim_iter = iter(sim_train)
            data = next(sim_iter)
        inputs_ids, attention_mask, token_type_ids, type_ids = move_to_device(
            data, device
        )
        with torch.autocast(
            device_type=device, enabled=config["fp16_run"], dtype=amp_type
        ):
            _, _, sim_hat = model(
                inputs_ids, attention_mask, token_type_ids, None, nsp=True
            )
        sim_loss = loss_func(sim_hat.float(), type_ids).mean()
        backward(sim_loss * config["c_sim"])

        if scaler is None:
            torch.nn.utils.clip_grad_norm_(model.bert.parameters(), config["max_grad"])
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.bert.parameters(), config["max_grad"])
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        global_step += 1
        if global_step % config["log_steps"] == 0:
            all_loss = (
                pdy_loss * config["c_pdy"]
                + emo_loss * config["c_emo"]
                + sim_loss * config["c_sim"]
            )
            logger.info(
                f"step={global_step} {all_loss=:.3f} {pdy_loss=:.3f} {emo_loss=:.3f} {sim_loss=:.3f}"
            )
            add_scalars(
                writer,
                {
                    "all_loss": all_loss,
                    "pdy_loss": pdy_loss,
                    "emo_loss": emo_loss,
                    "sim_loss": sim_loss,
                },
                global_step,
            )
        if global_step % config["eval_steps"] == 0:
            pdy_accu, emo_accu, sim_accu = eval(
                config, device, model, (pdy_valid, emo_valid, sim_valid)
            )
            all_accu = (
                pdy_accu * config["c_pdy"]
                + emo_accu * config["c_emo"]
                + sim_accu * config["c_sim"]
            )
            logger.info(
                f"step={global_step} {all_accu=:.3f} {pdy_accu=:.3f} {emo_accu=:.3f} {sim_accu=:.3f}"
            )
            add_scalars(
                writer,
                {
                    "all_accu": all_accu,
                    "pdy_accu": pdy_accu,
                    "emo_accu": emo_accu,
                    "sim_accu": sim_accu,
                },
                global_step,
            )
        if global_step % config["save_steps"] == 0:
            utils.save_checkpoint(
                model,
                optimizer,
                config["lr"],
                epoch,
                path.join(config["model_dir"], f"pdy-{global_step}.pth"),
            )


def eval(config, device, model: Model, loaders):
    model.eval()
    pdy_valid, emo_valid, sim_valid = loaders
    with torch.no_grad():
        correct_items, all_items = 0, 0
        for data in pdy_valid:
            inputs_ids, attention_mask, token_type_ids, token_ids, type_ids = (
                move_to_device(data, device)
            )
            pdy_hat, _, _ = model(inputs_ids, attention_mask, token_type_ids, token_ids)
            type_ids = type_ids.flatten()
            pdy_mask = type_ids != config["pdy_feats"]
            type_ids[~pdy_mask] = 0
            pdy_hat = torch.softmax(pdy_hat.reshape(type_ids.size(0), -1), dim=-1)
            pdy_hat = torch.argmax(pdy_hat, dim=-1)
            preds = (pdy_hat == type_ids) * pdy_mask
            correct_items += preds.sum().item()
            all_items += pdy_mask.sum().item()
        pdy_accu = correct_items / all_items

        correct_items, all_items = 0, 0
        for data in emo_valid:
            inputs_ids, attention_mask, token_type_ids, token_ids, type_ids = move_to_device(
                data, device
            )
            _, emo_hat, _ = model(
                inputs_ids, attention_mask, token_type_ids, token_ids, nsp=False
            )
            emo_hat = torch.softmax(emo_hat, dim=-1)
            emo_hat = torch.argmax(emo_hat, dim=-1)
            preds = emo_hat == type_ids
            correct_items += preds.sum().item()
            all_items += type_ids.size(0)
        emo_accu = correct_items / all_items

        correct_items, all_items = 0, 0
        for data in sim_valid:
            inputs_ids, attention_mask, token_type_ids, type_ids = move_to_device(
                data, device
            )
            _, _, sim_hat = model(
                inputs_ids, attention_mask, token_type_ids, None, nsp=True
            )
            sim_hat = torch.softmax(sim_hat, dim=-1)
            sim_hat = torch.argmax(sim_hat, dim=-1)
            preds = sim_hat == type_ids
            correct_items += preds.sum().item()
            all_items += type_ids.size(0)
        sim_accu = correct_items / all_items
    model.train()
    return pdy_accu, emo_accu, sim_accu


def move_to_device(data, device):
    res = ()
    for d in data:
        res += (d.to(device),)
    return res


def add_scalars(writer, scalars, step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, step)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "40000"
    main()
