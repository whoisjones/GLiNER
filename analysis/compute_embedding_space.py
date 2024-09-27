import sys

sys.path.append("/vol/fob-vol7/mi18/goldejon/GLiNER")

import copy
import glob
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data import create_dataset
from evaluate_gliner import configs
from evaluate_hierarchy import filter_neretrieve, hierarchy, relabel
from evaluate_synonyms import synonyms
from gliner import GLiNER
from gliner.modules.run_evaluation import inject_synonyms


def load_hierarchy_samples():
    if not os.path.exists(
        "/vol/tmp/goldejon/gliner/eval_embeddings/hierarchy_samples.json"
    ):
        with open(
            "/vol/tmp/goldejon/gliner/train_datasets/neretrieve_train.json", "r"
        ) as f:
            data = json.load(f)

        subsets = {}
        for coarse_type, fine_types in hierarchy.items():
            subset = []
            filtered_dataset = filter_neretrieve(data, coarse_type)
            random.shuffle(filtered_dataset)
            counter = Counter(fine_types)
            for sample in filtered_dataset:
                tags = Counter([y for x in sample["ner"] for y in x[-1]])
                matches = Counter(set(tags) & set(fine_types))
                if any([count > 101 for count in (matches + counter).values()]):
                    continue
                elif all([count == 101 for count in (matches + counter).values()]):
                    subset.append(sample)
                    break
                else:
                    counter = matches + counter
                    subset.append(sample)

            coarse_subset = copy.deepcopy(subset)
            fine_subset = copy.deepcopy(subset)
            coarse_subset, fine_subset = relabel(
                coarse_subset, fine_subset, coarse_type
            )
            subsets[coarse_type] = {}
            subsets[coarse_type]["top_level"] = {
                "dataset": coarse_subset,
                "entity_types": [coarse_type],
            }
            subsets[coarse_type]["subclass"] = {
                "dataset": fine_subset,
                "entity_types": fine_types,
            }

        with open(
            "/vol/tmp/goldejon/gliner/eval_embeddings/hierarchy_samples.json", "w"
        ) as f:
            json.dump(subsets, f)
    else:
        with open(
            "/vol/tmp/goldejon/gliner/eval_embeddings/hierarchy_samples.json", "r"
        ) as f:
            subsets = json.load(f)
    return subsets


def load_synonym_samples():
    if not os.path.exists(
        "/vol/tmp/goldejon/gliner/eval_embeddings/synonyms_samples.json"
    ):
        zero_shot_benc = [
            "mit-movie",
            "mit-restaurant",
            "CrossNER_AI",
            "CrossNER_literature",
            "CrossNER_music",
            "CrossNER_politics",
            "CrossNER_science",
        ]

        all_paths = glob.glob("/vol/tmp/goldejon/gliner/eval_datasets/NER/*")
        all_paths = [
            path
            for path in all_paths
            if "sample_" not in path and path.split("/")[-1] in zero_shot_benc
        ]

        counter = Counter(synonyms.keys())
        subsets = {}

        all_datasets = {}
        for path in all_paths:
            _, _, test_dataset, entity_types = create_dataset(path)

            test_datasets_iter, entity_types_iter = inject_synonyms(
                test_dataset, entity_types, synonyms
            )

            all_datasets[path.split("/")[-1]] = (test_datasets_iter, entity_types_iter)

        while not all([count == 101 for count in counter.values()]):

            _, (benchmark_iter, types_iter) = random.choice(list(all_datasets.items()))
            sampling = np.arange(len(benchmark_iter[0]))
            np.random.shuffle(sampling)

            for dp_idx in sampling:
                sample = benchmark_iter[0][dp_idx]
                types = [x[-1] for x in sample["ner"]]
                matches = Counter(
                    {k: v for k, v in Counter(types).items() if k in synonyms.keys()}
                )

                if any([count > 101 for count in (matches + counter).values()]):
                    continue

                if matches:
                    counter = matches + counter
                    for bm_idx, (bm, et) in enumerate(zip(benchmark_iter, types_iter)):
                        data_point = bm[dp_idx]
                        matched_types = list(matches.keys())
                        keys_to_check_with = [
                            _synonyms[bm_idx]
                            for original_label, _synonyms in synonyms.items()
                            if original_label in matched_types
                        ]
                        data_point["ner"] = [
                            annotation
                            for annotation in data_point["ner"]
                            if annotation[-1] in keys_to_check_with
                        ]
                        if bm_idx not in subsets:
                            subsets[bm_idx] = []
                        subsets[bm_idx].append(data_point)
                    break
                else:
                    continue

        formatted_subsets = []
        synonym_labels = list(zip(*synonyms.items()))
        for i in range(5):
            formatted_subsets.append(
                {"dataset": subsets[str(i)], "entity_types": synonym_labels[i]}
            )
        subsets = formatted_subsets

        with open(
            "/vol/tmp/goldejon/gliner/eval_embeddings/synonyms_samples.json", "w"
        ) as f:
            json.dump(subsets, f)
    else:
        with open(
            "/vol/tmp/goldejon/gliner/eval_embeddings/synonyms_samples.json", "r"
        ) as f:
            subsets = json.load(f)

    return subsets


def get_embeddings(model, dataset, entity_types, ft_dataset):
    inputs = []
    for sample in dataset:
        type_inputs = [
            tkn
            for tkn_lst in [
                [model.entity_token, entity_type] for entity_type in entity_types
            ]
            for tkn in tkn_lst
        ] + [model.sep_token]
        token_inputs = sample["tokenized_text"]
        inputs.append(
            {
                "tokens": type_inputs + token_inputs,
                "ner": sample["ner"],
                "class_to_id": {
                    label: idx + 1 for idx, label in enumerate(entity_types)
                },
                "label_length": len(type_inputs),
                "token_length": len(token_inputs),
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = {"embedding": [], "type": [], "value": [], "dataset": []}
    for input in inputs:
        span_idx = []
        for i in range(input["token_length"]):
            span_idx.extend([(i, i + j) for j in range(model.max_width)])
        dict_lab = (
            get_dict(input["ner"], input["class_to_id"])
            if input["ner"]
            else defaultdict(int)
        )

        span_label = torch.LongTensor([dict_lab[i] for i in span_idx]).to(device)
        span_idx = torch.LongTensor(span_idx).to(device)

        valid_span_mask = span_idx[:, 1] < input["token_length"]
        span_idx = span_idx * valid_span_mask.unsqueeze(-1)

        outputs = model.token_rep_layer(
            [input["tokens"]],
            torch.tensor([input["token_length"] + input["label_length"]]),
        )
        entity_rep, token_rep = (
            outputs["embeddings"][:, : input["label_length"] - 1],
            outputs["embeddings"][:, input["label_length"] :],
        )
        entity_rep = entity_rep[:, 0::2]
        entity_rep = model.prompt_rep_layer(entity_rep).squeeze(0)
        word_rep = model.rnn(token_rep, outputs["mask"][:, input["label_length"] :])
        span_rep = model.span_rep_layer(word_rep, span_idx.unsqueeze(0))
        span_rep = span_rep.view(1, input["token_length"] * model.max_width, 768)
        mention_mask = (span_label != 0).unsqueeze(0)
        mention_reps = span_rep[mention_mask]

        for idx, annotation in enumerate(input["ner"]):
            df["embedding"].append(mention_reps[idx].cpu().detach().numpy())
            df["type"].append("mention")
            df["value"].append(annotation[-1])

        labels_in_sentence = set([label for _, _, label in input["ner"]])
        for label, idx in input["class_to_id"].items():
            if label in labels_in_sentence:
                df["embedding"].append(entity_rep[idx - 1].cpu().detach().numpy())
                df["type"].append("entity")
                df["value"].append(label)

    df["dataset"] = [ft_dataset] * len(df["embedding"])

    return pd.DataFrame.from_dict(df)


def get_dict(spans, classes_to_id):
    dict_tag = defaultdict(int)
    for span in spans:
        if span[2] in classes_to_id:
            dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
    return dict_tag


def embed_hierarchy(model: torch.nn.Module, datasets: Dict, ft_dataset: str, seed: int):
    for _, hierarchy_datasets in datasets.items():
        top_level_dataset = hierarchy_datasets["top_level"]["dataset"]
        top_level_entity_types = hierarchy_datasets["top_level"]["entity_types"]
        subclass_dataset = hierarchy_datasets["subclass"]["dataset"]
        subclass_entity_types = hierarchy_datasets["subclass"]["entity_types"]

        top_level_embeddings = get_embeddings(
            model, top_level_dataset, top_level_entity_types, ft_dataset
        )
        top_level_embeddings["seed"] = [seed] * top_level_embeddings.shape[0]
        top_level_embeddings["label_type"] = ["top_level"] * top_level_embeddings.shape[
            0
        ]

        subclass_embeddings = get_embeddings(
            model, subclass_dataset, subclass_entity_types, ft_dataset
        )
        subclass_embeddings["seed"] = [seed] * subclass_embeddings.shape[0]
        subclass_embeddings["label_type"] = ["subclass"] * subclass_embeddings.shape[0]

        return pd.concat([top_level_embeddings, subclass_embeddings], ignore_index=True)


def embed_synonyms(model: torch.nn.Module, datasets: Dict, ft_dataset: str, seed: int):
    embeddings = pd.DataFrame()
    for idx, dataset in enumerate(datasets):
        _embeddings = get_embeddings(
            model, dataset["dataset"], dataset["entity_types"], ft_dataset
        )
        _embeddings["seed"] = [seed] * _embeddings.shape[0]
        _embeddings["label_type"] = [
            "original" if idx == 0 else "synonym"
        ] * _embeddings.shape[0]
        embeddings = pd.concat([embeddings, _embeddings], ignore_index=True)
    return embeddings


def compute_tsne(embeddings):
    tsne = TSNE(n_components=2, n_iter=10000)
    embeddings_tsne = tsne.fit_transform(np.stack(embeddings["embedding"]))
    embeddings_tsne = MinMaxScaler().fit_transform(embeddings_tsne)
    x, y = embeddings_tsne[:, 0], embeddings_tsne[:, 1]
    x = np.split(x, np.arange(1, len(x)))
    y = np.split(y, np.arange(1, len(y)))
    embeddings["x"] = x
    embeddings["y"] = y
    embeddings["x"] = embeddings["x"].astype(float)
    embeddings["y"] = embeddings["y"].astype(float)
    embeddings.drop("embedding", axis=1, inplace=True)
    return embeddings


def compute_umap(embeddings):
    reducer = umap.UMAP()
    embeddings_umap = reducer.fit_transform(np.stack(embeddings["embedding"]))
    embeddings_umap = MinMaxScaler().fit_transform(embeddings_umap)
    x, y = embeddings_umap[:, 0], embeddings_umap[:, 1]
    x = np.split(x, np.arange(1, len(x)))
    y = np.split(y, np.arange(1, len(y)))
    embeddings["x"] = x
    embeddings["y"] = y
    embeddings["x"] = embeddings["x"].astype(float)
    embeddings["y"] = embeddings["y"].astype(float)
    embeddings.drop("embedding", axis=1, inplace=True)
    return embeddings


def plot_tsne(embeddings):
    tsne_embeddings = pd.DataFrame()
    for seed in embeddings["seed"].unique():
        for dataset in embeddings["dataset"].unique():
            for _, synonym_labels in synonyms.items():
                filtered_embeddings = embeddings[embeddings["seed"] == seed]
                filtered_embeddings = filtered_embeddings[
                    filtered_embeddings["dataset"] == dataset
                ]
                filtered_embeddings = filtered_embeddings[
                    filtered_embeddings["value"].isin(synonym_labels)
                ]
                filtered_embeddings = filtered_embeddings[
                    filtered_embeddings["type"] == "entity"
                ]
                tsne_dataset_seed = compute_umap(filtered_embeddings)
                tsne_embeddings = pd.concat(
                    [tsne_embeddings, tsne_dataset_seed], ignore_index=True
                )

    for seed in tsne_embeddings["seed"].unique():
        tsne_embeddings_seed = tsne_embeddings[tsne_embeddings["seed"] == seed]

        sns.set_theme(style="darkgrid", font_scale=1)
        g = sns.displot(
            tsne_embeddings_seed,
            x="x",
            y="y",
            col="dataset",
            hue="label_type",
            kind="kde",
            col_wrap=3,
            levels=7,
            alpha=0.8,
        )
        g.savefig(f"umap_minmax_synonyms_{seed}.png")


if __name__ == "__main__":
    random.seed(123)
    hierarchy_samples = load_hierarchy_samples()
    synonyms_samples = load_synonym_samples()
    if not os.path.exists("/vol/tmp/goldejon/gliner/eval_embeddings/embeddings.pkl"):
        embeddings = pd.DataFrame()
        for ft_dataset, model_paths in configs.items():
            for model_path in model_paths:
                model_ckpt = f"/vol/tmp/goldejon/gliner/logs/deberta-v3-large/{ft_dataset}/{model_path}"
                seed, _ = model_path.split("/")

                model = GLiNER.from_pretrained(model_ckpt)
                model.eval()
                if torch.cuda.is_available():
                    model = model.to("cuda")

                hierarchy_embeddings = embed_hierarchy(
                    model, hierarchy_samples, ft_dataset, seed
                )
                synonyms_embeddings = embed_synonyms(
                    model, synonyms_samples, ft_dataset, seed
                )
                embeddings = pd.concat(
                    [embeddings, hierarchy_embeddings, synonyms_embeddings],
                    ignore_index=True,
                )

        embeddings.to_pickle("/vol/tmp/goldejon/gliner/eval_embeddings/embeddings.pkl")
    else:
        embeddings = pd.read_pickle(
            "/vol/tmp/goldejon/gliner/eval_embeddings/embeddings.pkl"
        )

    plot_tsne(embeddings)
