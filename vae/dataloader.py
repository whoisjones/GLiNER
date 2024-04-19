import random
from collections import defaultdict
from typing import List, Dict, Tuple

from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class VAEDataLoader:
    def __init__(
        self,
        token_encoder: str,
        label_encoder: str,
        dataloader_config: Dict,
        device: str,
    ):
        super().__init__()
        self.dataloader_config = dataloader_config

        self.ent_token = "[ENT]"

        self.label_tokenizer = AutoTokenizer.from_pretrained(label_encoder)
        self.label_tokenizer.add_tokens([self.ent_token])
        self.token_tokenizer = AutoTokenizer.from_pretrained(token_encoder)

        self.device = device

    def get_train_loader(self, data: List[Dict], **kwargs):
        return DataLoader(
            data,
            collate_fn=lambda x: self.collate_fn(
                x,
                dataloader_config=self.dataloader_config,
            ),
            **kwargs
        )

    def get_validation_loader(
        self, data: List[Dict], entity_types: List[str], **kwargs
    ):
        return DataLoader(
            data,
            collate_fn=lambda example: self.collate_fn(
                example,
                dataloader_config=self.dataloader_config,
                entity_types=entity_types,
            ),
            **kwargs
        )

    def collate_fn(
        self,
        raw_inputs: List[Dict[str, List]],
        dataloader_config: Dict,
        entity_types: List[str] = None,
    ):
        if entity_types is None:
            batch = self.make_training_batch(raw_inputs, dataloader_config)
        else:
            batch = self.make_validation_batch(
                raw_inputs, dataloader_config, entity_types
            )

        return batch

    def make_training_batch(
        self,
        raw_inputs: List[Dict[str, List]],
        dataloader_config: Dict,
    ) -> List:
        class_to_ids = self.sample_annotations(raw_inputs, dataloader_config)

        raw_batch = [
            self.preprocess_data_point(
                tokens=data_point["tokenized_text"],
                ner=data_point["ner"],
                classes_to_id=class_to_ids[i],
                max_len=dataloader_config.get("max_len"),
                max_width=dataloader_config.get("max_width"),
                learn_only_positives=dataloader_config.get("learn_only_positives"),
            )
            for i, data_point in enumerate(raw_inputs)
        ]

        batch = self.batch_format(raw_batch)

        return batch

    def sample_annotations(
        self,
        raw_inputs: List[Dict[str, List]],
        dataloader_config: Dict,
    ):
        negs = self.sample_in_batch_negatives(raw_inputs, 100)
        class_to_ids = []

        for data_point in raw_inputs:
            random.shuffle(negs)

            max_neg_type_ratio = int(dataloader_config.get("max_neg_type_ratio"))

            if max_neg_type_ratio == 0:
                # no negatives
                neg_type_ratio = 0
            else:
                neg_type_ratio = random.randint(0, max_neg_type_ratio)

            if neg_type_ratio == 0:
                # no negatives
                negs_i = []
            else:
                negs_i = negs[: len(data_point["ner"]) * neg_type_ratio]

            # this is the list of all possible entity types (positive and negative)
            types = list(set([el[-1] for el in data_point["ner"]] + negs_i))

            # shuffle (every epoch)
            random.shuffle(types)

            if len(types) != 0:
                # prob of higher number shoul
                # random drop
                if dataloader_config.get("random_drop"):
                    num_ents = random.randint(1, len(types))
                    types = types[:num_ents]

            # maximum number of entities types
            types = types[: int(dataloader_config.get("max_types"))]

            # supervised training
            if "label" in data_point:
                types = sorted(data_point["label"])

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            class_to_ids.append(class_to_id)

        return class_to_ids

    def sample_in_batch_negatives(self, batch_list, sampled_neg=5):
        ent_types = []
        for b in batch_list:
            types = set([el[-1] for el in b["ner"]])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def sample_litset_in_batch_negatives(self, batch_list, sampled_neg=5):
        ent_types = []
        for b in batch_list:
            descriptions = set(
                [el[-1]["description"][0] for el in b["ner"] if el[-1]["description"]]
            )
            labels = set(
                [
                    flat_labels
                    for l in [el[-1]["labels"] for el in b["ner"]]
                    for flat_labels in l
                ]
            )
            ent_types.extend(list(descriptions))
            ent_types.extend(list(labels))
        ent_types = list(set(ent_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def make_validation_batch(
        self,
        raw_inputs: List[Dict[str, List]],
        dataloader_config: Dict,
        entity_types: List[str] = None,
    ) -> List:
        class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}

        raw_batch = [
            self.preprocess_data_point(
                tokens=data_point["tokenized_text"],
                ner=data_point["ner"],
                classes_to_id=class_to_ids,
                max_len=dataloader_config.get("max_len"),
                max_width=dataloader_config.get("max_width"),
                learn_only_positives=dataloader_config.get("learn_only_positives"),
            )
            for data_point in raw_inputs
        ]

        batch = self.batch_format(raw_batch)

        return batch

    def preprocess_data_point(
        self,
        tokens: List[str],
        ner: List[Tuple[int, int, str]],
        classes_to_id: Dict,
        max_len: int = 384,
        max_width: int = 12,
        learn_only_positives: bool = False,
    ):
        if len(tokens) > max_len:
            length = max_len
            tokens = tokens[:max_len]
        else:
            length = len(tokens)

        spans_idx = []
        for i in range(length):
            spans_idx.extend([(i, i + j) for j in range(max_width)])

        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)

        # 0 for null labels
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)

        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > length - 1

        if learn_only_positives:
            non_entity_mask = span_label == 0
            valid_span_mask = torch.logical_or(valid_span_mask, non_entity_mask)

        # mask invalid positions
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            "tokens": tokens,
            "class_to_id": classes_to_id,
            "id_to_class": {v: k for k, v in classes_to_id.items()},
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": length,
            "entities": ner,
        }

    def get_dict(self, spans, classes_to_id):
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    def batch_format(self, raw_batch: List[Dict]):

        # Converts to list of inputs
        tokens = [
            [self.token_tokenizer.sep_token]
            + data_point["tokens"]
            + [self.token_tokenizer.sep_token]
            for data_point in raw_batch
        ]

        token_batch = self.token_tokenizer(
            tokens,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        token_lengths = torch.sum(token_batch["attention_mask"], dim=1)

        word_ids = [
            self.filter_nones_and_special_tokens(
                token_batch.word_ids(i), token_batch["input_ids"][i]
            )
            for i in range(len(raw_batch))
        ]
        word_lengths = [len(seq) - 2 for seq in tokens]
        word_rep_mask = torch.arange(max(word_lengths)).unsqueeze(0).expand(
            len(raw_batch), -1
        ) < torch.tensor(word_lengths).unsqueeze(-1)

        entities = [data_point["entities"] for data_point in raw_batch]
        class_to_ids = [data_point["class_to_id"] for data_point in raw_batch]
        id_to_classes = [data_point["id_to_class"] for data_point in raw_batch]

        # Create list of labels for each input
        labels = []
        label_lengths = []
        for class_to_id in class_to_ids:
            entity_types = list(class_to_id.keys())
            label_lengths.append(len(entity_types))
            labels.append(
                [
                    inner
                    for outer in [
                        [entity_type, self.ent_token] for entity_type in entity_types
                    ]
                    for inner in outer
                ]
            )
        label_lengths = torch.LongTensor(label_lengths).to(self.device)

        label_batch = self.label_tokenizer(
            labels,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        ).to(self.device)

        label_mask = label_batch[
            "input_ids"
        ] == self.label_tokenizer.convert_tokens_to_ids(self.ent_token)

        # Input lengths are the sum of token lengths and prompt lengths
        combined_lengths = token_lengths + label_lengths

        span_idx = pad_sequence(
            [data_point["span_idx"] for data_point in raw_batch],
            batch_first=True,
            padding_value=0,
        )

        span_label = pad_sequence(
            [data_point["span_label"] for data_point in raw_batch],
            batch_first=True,
            padding_value=-1,
        )

        span_mask = span_label != -1

        span_idx = span_idx * span_mask.unsqueeze(-1)

        # create a mask using num_classes_all (0, if it exceeds the number of classes, 1 otherwise)
        num_classes = [len(classes) for classes in class_to_ids]
        max_num_classes = max(num_classes)
        entity_type_mask = (
            torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes), -1)
        )
        entity_type_mask = entity_type_mask < torch.tensor(num_classes).unsqueeze(-1)

        return {
            "token_inputs": token_batch,
            "label_inputs": label_batch,
            "label_mask": label_mask,
            "label_lengths": label_lengths,
            "token_lengths": token_lengths,
            "combined_lengths": combined_lengths,
            "word_ids": word_ids,
            "word_rep_mask": word_rep_mask,
            "span_idx": span_idx,
            "span_label": span_label,
            "entity_type_mask": entity_type_mask,
            "entities": entities,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
            "num_classes": num_classes,
        }

    def filter_nones_and_special_tokens(self, seq, input_ids):
        return [
            word_id
            for word_id, input_id in zip(seq, input_ids)
            if word_id is not None and input_id != self.token_tokenizer.sep_token_id
        ]
