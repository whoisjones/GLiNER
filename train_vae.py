import os
import json
import random
import datetime
from tqdm import tqdm

import torch
import numpy as np

from transformers import get_cosine_schedule_with_warmup
from train import create_parser, load_config_as_namespace, sample_train_data

from vae.config import VAEConfig
from vae.model import VAEModel
from vae.dataloader import VAEDataLoader


def train(
    model,
    optimizer,
    train_data,
    num_steps=1000,
    eval_every=100,
    log_dir="logs",
    warmup_ratio=0.1,
    train_batch_size=8,
    device="cuda",
):
    model.train()

    data_loader = VAEDataLoader(
        token_encoder=config.token_encoder,
        label_encoder=config.label_encoder,
        dataloader_config=config.dataloader,
        device=device,
    )

    train_loader = data_loader.get_train_loader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
    )

    pbar = tqdm(range(num_steps))

    if warmup_ratio < 1:
        num_warmup_steps = int(num_steps * warmup_ratio)
    else:
        num_warmup_steps = int(warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        try:
            outputs = model(x)  # Forward pass
        except:
            continue

        # check if loss is nan
        if torch.isnan(outputs["loss"]):
            continue

        loss = outputs["loss"]
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f} | vae: {outputs['vae_loss'].item():.2f}"

        if (step + 1) % eval_every == 0:
            current_path = os.path.join(log_dir, f"model_{step + 1}")
            model.save_pretrained(current_path)
            model.train()

        pbar.set_description(description)


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    parser = create_parser()
    args = parser.parse_args()
    config = load_config_as_namespace(args.config)

    dataset_log = config.train_data.split("/")[-1].replace(".json", "")
    config.log_dir = os.path.join(
        config.root_dir,
        config.log_dir,
        dataset_log,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    try:
        if "litset" in config.train_data:
            with open(
                os.path.join(config.root_dir, "train_datasets/litset.jsonl"), "r"
            ) as f:
                data = []
                for line in f.readlines():
                    data.append(json.loads(line))
                data = [
                    x for x in data if x["ner"]
                ]  # Comment this out if you want to use the entire dataset
        else:
            with open(os.path.join(config.root_dir, config.train_data), "r") as f:
                data = json.load(f)
    except:
        data = sample_train_data(config.train_data, 10000)

    vae_config = VAEConfig(**vars(config))
    vae_model = VAEModel(vae_config)

    if torch.cuda.is_available():
        vae_model = vae_model.cuda()

    lr_encoders = float(config.lr_encoders)
    lr_others = float(config.lr_others)

    optimizer = vae_model.get_optimizer(lr_encoders, lr_others)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        vae_model,
        optimizer,
        train_data=data,
        num_steps=config.num_steps,
        eval_every=config.eval_every,
        log_dir=config.log_dir,
        warmup_ratio=config.warmup_ratio,
        train_batch_size=config.train_batch_size,
        device=device,
    )
