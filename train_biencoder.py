import copy
import random
from argparse import ArgumentParser
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class BiEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenized_labels: BatchEncoding,
    ):
        super(BiEncoder, self).__init__()
        self.encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.decoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_config(self.encoder_config)
        self.decoder = AutoModel.from_config(self.decoder_config)
        self.tokenized_labels = tokenized_labels
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        labels: torch.tensor,
    ):
        token_hidden_states = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        label_inputs = copy.deepcopy(self.tokenized_labels).to(input_ids.device)
        label_hidden_states = self.decoder(**label_inputs)
        label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]
        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)

        if labels is not None:
            loss = self.loss(logits.transpose(1, 2), labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def load_dataset(dataset_name: str) -> List[Dict]:
    pass


def get_possible_data_points(dataset: Dataset, label_ids: List[int]):
    possible_data_points = {}
    for idx, example in enumerate(tqdm(dataset)):
        tags = [x for x in set(example["fine_ner_tags"]) if x != 0]
        if len(tags) == 1:
            tag = tags[0]
            if tag in label_ids:
                if tag not in possible_data_points:
                    possible_data_points[tag] = []
                possible_data_points[tag].append(idx)
    return possible_data_points


def downsample(
    dataset: DatasetDict,
    sampled_label_ids: List[int],
    max_train_samples_per_label: int,
    max_val_samples_per_label: int,
):
    possible_data_points = get_possible_data_points(dataset["train"], sampled_label_ids)
    all_train_data_points = []
    for data_points in possible_data_points.values():
        random.shuffle(data_points)
        all_train_data_points.extend(data_points[:max_train_samples_per_label])
    dataset["train"] = dataset["train"].select(all_train_data_points)

    possible_data_points = get_possible_data_points(
        dataset["validation"], sampled_label_ids
    )
    all_val_data_points = []
    for data_points in possible_data_points.values():
        random.shuffle(data_points)
        all_val_data_points.extend(data_points[:max_val_samples_per_label])
    dataset["validation"] = dataset["validation"].select(all_val_data_points)

    return dataset


def train_biencoder(args):
    if isinstance(args.num_labels, int):
        num_labels_sweep = [args.num_labels]
    else:
        num_labels_sweep = args.num_labels

    for num_labels in num_labels_sweep:
        random.seed(42)

        max_train_samples_per_label = 100
        max_val_samples_per_label = 10

        raw_datasets = load_dataset(dataset, config)
        label_names = [
            "I-" + tag if tag != "O" else tag
            for tag in raw_datasets["train"].features["fine_ner_tags"].feature.names
        ]
        sampled_label_ids = random.sample(range(1, len(label_names) + 1), num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

        dataset = downsample(
            raw_datasets,
            sampled_label_ids,
            max_train_samples_per_label,
            max_val_samples_per_label,
        )

        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    new_labels.append(label)

            return new_labels

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            all_labels = examples["fine_ner_tags"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        metric = evaluate.load("seqeval", scheme="IOB1")

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [
                [label_names[l] for l in label if l != -100] for label in labels
            ]
            true_predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metric.compute(
                predictions=true_predictions,
                references=true_labels,
            )
            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
            }

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        tokenized_labels = tokenizer(
            label_names, truncation=True, padding=True, return_tensors="pt"
        )

        model = BiEncoder(args.model_checkpoint, tokenized_labels)

        output_dir = f"ablation/dualencoder/{num_labels}"
        training_args = TrainingArguments(
            output_dir,
            save_strategy="steps",
            learning_rate=3e-5,
            max_steps=1000,
            weight_decay=0.01,
            warmup_ratio=0.1,
            push_to_hub=False,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_total_limit=1,
            fp16=True,
            metric_for_best_model="eval_f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            eval_accumulation_steps=500,
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()

        eval_results = trainer.evaluate(tokenized_datasets["validation"])
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)

        test_results = trainer.predict(tokenized_datasets["test"])
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)

        model.encoder_config.save_pretrained(output_dir + "/encoder_config")
        model.decoder_config.save_pretrained(output_dir + "/decoder_config")
        model.encoder.save_pretrained(output_dir + "/encoder")
        model.decoder.save_pretrained(output_dir + "/decoder")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="biencoder")
    parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--num_sentences", nargs="+", type=int, default=3000)
    parser.add_argument("--training_dataset", type=str, default="nuner")
    arguments = parser.parse_args()
    if arguments.model == "biencoder":
        train_biencoder(arguments)
    else:
        train_gliner(arguments)
