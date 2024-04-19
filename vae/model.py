from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    BatchEncoding,
)

from .config import VAEConfig
from modules.layers import LstmSeq2SeqEncoder
from modules.span_rep import SpanRepLayer


class LabelEncoder(nn.Module):
    def __init__(
        self, name_or_path: str, latent_dimension: int = 128, hidden_size: int = 768
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(name_or_path)
        self.model = AutoModel.from_config(self.config)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.mu = nn.Linear(hidden_size, latent_dimension)
        self.sigma = nn.Linear(hidden_size, latent_dimension)
        self.up_proj = nn.Linear(latent_dimension, hidden_size)
        self.re_proj = nn.Linear(hidden_size, hidden_size)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc.cpu()
            self.N.scale = self.N.scale.cpu()

    def forward(self, inputs: BatchEncoding):
        outputs = self.tanh(self.model(**inputs)[0])

        mu = self.mu(outputs)
        sigma = torch.exp(self.sigma(outputs))
        z = mu + sigma * self.N.sample(mu.shape)
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()

        outputs = self.relu(self.up_proj(z))
        embeddings = self.sigmoid(self.re_proj(outputs))
        return embeddings, kl_loss


class VAEModel(PreTrainedModel):
    config_class = VAEConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_encoder = LabelEncoder(config.label_encoder, config.latent_dimension)
        token_encoder_config = AutoConfig.from_pretrained(config.token_encoder)
        self.token_encoder = AutoModel.from_config(token_encoder_config)

        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.dataloader.get("max_width"),
            dropout=config.dropout,
        )

        self.prompt_rep_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def get_optimizer(self, lr_encoders, lr_others):
        param_groups = [
            {"params": self.token_encoder.parameters(), "lr": lr_encoders},
            {"params": self.label_encoder.parameters(), "lr": lr_encoders},
            {"params": self.rnn.parameters(), "lr": lr_others},
            {"params": self.span_rep_layer.parameters(), "lr": lr_others},
            {"params": self.prompt_rep_layer.parameters(), "lr": lr_others},
        ]

        optimizer = torch.optim.AdamW(param_groups)

        return optimizer

    def forward(self, batch):
        # compute span representation
        scores, kl_loss = self.compute_score_train(batch)
        batch_size = scores.shape[0]
        num_classes = batch["num_classes"]

        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)
        labels = batch["span_label"].view(-1)  # (batch_size * num_spans)
        mask_label = labels != -1  # (batch_size * num_spans)
        labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

        # one-hot encoding
        labels_one_hot = torch.zeros(
            labels.size(0), num_classes + 1, dtype=torch.float32
        ).to(scores.device)
        labels_one_hot.scatter_(
            1, labels.unsqueeze(1), 1
        )  # Set the corresponding index to 1
        labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column
        # Shape of labels_one_hot: (batch_size * num_spans, num_classes)

        # compute loss (without reduction)
        try:
            all_losses = F.binary_cross_entropy_with_logits(
                logits_label, labels_one_hot, reduction="none"
            )
        except:
            raise ValueError("Error in computing loss")
        # mask loss using entity_type_mask (B, C)
        masked_loss = all_losses.view(batch_size, -1, num_classes) * batch[
            "entity_type_mask"
        ].unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)
        # expand mask_label to all_losses
        mask_label = mask_label.unsqueeze(-1).expand_as(all_losses)
        # put lower loss for in label_one_hot (2 for positive, 1 for negative)
        weight_c = labels_one_hot + 1
        # apply mask
        all_losses = all_losses * mask_label.float() * weight_c
        return all_losses.sum() + kl_loss

    def compute_score_train(self, batch):
        label_latents, kl_loss = self.label_encoder(batch["label_inputs"])

        token_embeddings = self.token_encoder.embeddings(
            batch["token_inputs"]["input_ids"]
        )

        batch_size, seq_len, hidden_size = token_embeddings.shape

        combined_inputs = torch.zeros(
            batch_size, torch.max(batch["combined_lengths"]), hidden_size
        ).to(token_embeddings.device)

        combined_attention_mask = torch.zeros(
            batch_size, torch.max(batch["combined_lengths"])
        ).to(token_embeddings.device)

        for i in range(batch_size):
            latent_label = label_latents[i][batch["label_mask"][i]]
            combined_inputs[i, : batch["label_lengths"][i]] = latent_label
            combined_attention_mask[i, : batch["label_lengths"][i]] = 1

            combined_inputs[
                i, batch["label_lengths"][i] : batch["combined_lengths"][i]
            ] = token_embeddings[i, : batch["token_lengths"][i]]

            combined_attention_mask[
                i, batch["label_lengths"][i] : batch["combined_lengths"][i]
            ] = 1

        outputs = self.token_encoder.encoder(combined_inputs, combined_attention_mask)[
            0
        ]

        label_outputs = pad_sequence(
            [outputs[i, : batch["label_lengths"][i]] for i in range(batch_size)],
            batch_first=True,
        )

        word_outputs = [
            outputs[i, batch["label_lengths"][i] + 1 : batch["combined_lengths"][i] - 1]
            for i in range(batch_size)
        ]

        word_rep = pad_sequence(
            self.subtoken_pooling(word_outputs, batch["word_ids"]), batch_first=True
        )

        # compute span representation
        word_rep = self.rnn(word_rep, batch["word_rep_mask"])
        span_rep = self.span_rep_layer(word_rep, batch["span_idx"])

        # compute final entity type representation (FFN)
        entity_type_rep = self.prompt_rep_layer(label_outputs)

        # similarity score
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, entity_type_rep)

        return scores, kl_loss

    def subtoken_pooling(self, word_outputs: List, word_ids: List, how: str = "first"):
        # sanity check
        assert [x.size(0) for x in word_outputs] == [len(w) for w in word_ids]

        pooled_word_reps = []
        for word_rep, word_id in zip(word_outputs, word_ids):
            current = []

            if how == "first":
                previous_idx = None
                for rep, idx in zip(word_rep, word_id):
                    if idx != previous_idx:
                        current.append(rep)
                        previous_idx = idx

            elif how == "mean":
                word = []
                previous_idx = None
                for rep, idx in zip(word_rep, word_id):
                    if idx != previous_idx and word:
                        current.append(torch.stack(word).mean(0))
                        word = [rep]
                        previous_idx = idx
                    elif idx == previous_idx:
                        word.append(rep)
                    else:
                        word = [rep]
                        previous_idx = idx

            else:
                raise ValueError(f"Invalid pooling method: {how}")

            pooled_word_reps.append(torch.stack(current))

        return pooled_word_reps
