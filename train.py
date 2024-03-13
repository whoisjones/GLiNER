import argparse
import os
import datetime

import torch
import yaml
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# from model_nested import NerFilteredSemiCRF
from model import GLiNER
from modules.run_evaluation import get_for_all_path, sample_train_data
from save_load import save_model, load_model
import json


# train function
def train(
    model,
    optimizer,
    train_data,
    eval_data=None,
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
        train_data, batch_size=train_batch_size, shuffle=True
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

        if (step + 1) % eval_every == 0:
            current_path = os.path.join(log_dir, f"model_{step + 1}")
            save_model(model, current_path)
            model.train()

        pbar.set_description(description)

    if eval_data:
        get_for_all_path(model, step, log_dir, eval_data)


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    return parser


def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    # load config
    config = load_config_as_namespace(args.config)

    model_name_log = config.model_name.split("/")[-1]
    dataset_log = config.train_data.split("/")[-1].replace(".json", "")
    config.log_dir = os.path.join(
        config.root_dir,
        config.log_dir,
        model_name_log,
        dataset_log,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    try:
        with open(os.path.join(config.root_dir, config.train_data), "r") as f:
            data = json.load(f)
    except:
        data = sample_train_data(config.train_data, 10000)

    if config.do_eval:
        val_data_dir = os.path.join(config.root_dir, config.eval_data)

    if config.prev_path != "none":
        model = load_model(config.prev_path)
        model.config = config
    else:
        model = GLiNER(config)

    if torch.cuda.is_available():
        model = model.cuda()

    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)

    optimizer = torch.optim.AdamW(
        [
            # encoder
            {"params": model.token_rep_layer.parameters(), "lr": lr_encoder},
            {"params": model.rnn.parameters(), "lr": lr_others},
            # projection layers
            {"params": model.span_rep_layer.parameters(), "lr": lr_others},
            {"params": model.prompt_rep_layer.parameters(), "lr": lr_others},
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        model,
        optimizer,
        train_data=data,
        eval_data=val_data_dir if config.do_eval else None,
        num_steps=config.num_steps,
        eval_every=config.eval_every,
        log_dir=config.log_dir,
        warmup_ratio=config.warmup_ratio,
        train_batch_size=config.train_batch_size,
        device=device,
    )
