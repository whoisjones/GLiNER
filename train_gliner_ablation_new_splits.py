import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# from model_nested import NerFilteredSemiCRF
from gliner import GLiNER
from gliner.model import load_config_as_namespace
from gliner.modules.run_evaluation import get_for_all_path
from compute_new_splits import create_splits


# train function
def train(
    model,
    optimizer,
    train_data,
    dataset_name: str,
    num_steps=1000,
    eval_every=100,
    log_dir="logs",
    warmup_ratio=0.1,
    train_batch_size=8,
    device="cuda",
):
    model.train()

    # initialize data loaders
    train_loader = model.create_dataloader(
        train_data, dataset_name=dataset_name, batch_size=train_batch_size, shuffle=True
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
            loss = model(x)  # Forward pass
        except:
            continue

        # check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        del loss

        pbar.set_description(description)

    model.save_pretrained(os.path.join(log_dir, f"model_{num_steps}"))
    val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
    get_for_all_path(model, log_dir, data_paths=val_data_dir)


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument(
        "--config", type=str, default="config_large.yaml", help="Path to config file"
    )
    parser.add_argument("--train_dataset", type=str)
    return parser


if __name__ == "__main__":
    log_dir = "/vol/tmp/goldejon/gliner/logs_ablation_new_splits/{dataset}/{setting}/{filter_by}/{seed}"
    train_data_dir = "/vol/tmp/goldejon/gliner/train_datasets/{dataset}.json"
    for filter_by in ["entropy", "max"]:
        for setting in ["easy", "medium", "hard"]:
            for seed in [123]:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                # parse args
                parser = create_parser()
                args = parser.parse_args()

                # load config
                config = load_config_as_namespace(args.config)

                try:
                    train_dataset_path = train_data_dir.format(
                        dataset=args.train_dataset
                    )
                    with open(train_dataset_path, "r") as f:
                        data = json.load(f)

                        data = create_splits(
                            data,
                            args.train_dataset,
                            filter_by=filter_by,
                            setting=setting,
                        )

                    if len(data) > config.train_batch_size * config.num_steps:
                        data = random.sample(
                            data, config.train_batch_size * config.num_steps
                        )
                except:
                    raise ValueError("Invalid data path")

                model_name = config.model_name.split("/")[-1]
                run_dir = log_dir.format(
                    model=model_name,
                    dataset=args.train_dataset,
                    seed=seed,
                    setting=setting,
                    filter_by=filter_by,
                )
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)
                else:
                    raise ValueError("Log directory already exists")

                if config.prev_path != "none":
                    model = GLiNER.from_pretrained(config.prev_path)
                    model.config = config
                else:
                    model = GLiNER(config)

                if torch.cuda.is_available():
                    model = model.to("cuda")

                lr_encoder = float(config.lr_encoder)
                lr_others = float(config.lr_others)

                optimizer = torch.optim.AdamW(
                    [
                        # encoder
                        {
                            "params": model.token_rep_layer.parameters(),
                            "lr": lr_encoder,
                        },
                        {"params": model.rnn.parameters(), "lr": lr_others},
                        # projection layers
                        {"params": model.span_rep_layer.parameters(), "lr": lr_others},
                        {
                            "params": model.prompt_rep_layer.parameters(),
                            "lr": lr_others,
                        },
                    ]
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"

                train(
                    model,
                    optimizer,
                    data,
                    dataset_name=args.train_dataset,
                    num_steps=config.num_steps,
                    eval_every=config.eval_every,
                    log_dir=run_dir,
                    warmup_ratio=config.warmup_ratio,
                    train_batch_size=config.train_batch_size,
                    device=device,
                )
