import logging

import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from save_load import load_model
from evaluate_synonyms import synonyms
from evaluate_intersections import labels

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter with timestamp
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create console handler and set level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)


def analyze_embedding_space(model, dataset):
    model.eval()
    inputs = []
    lengths = []
    labels_synonyms = []
    label_types = []
    for original_label, synonym_labels in synonyms.items():
        inputs.append([model.entity_token, original_label])
        lengths.append(len(inputs[-1]))
        labels_synonyms.append("original")
        label_types.append(original_label)
        for synonym_label in synonym_labels:
            if synonym_label == original_label:
                continue
            inputs.append([model.entity_token, synonym_label])
            lengths.append(len(inputs[-1]))
            labels_synonyms.append("synonym")
            label_types.append(original_label)

    labels_intersections = []
    for original_label, intersection_label in labels.items():
        inputs.append([model.entity_token, original_label])
        lengths.append(len(inputs[-1]))
        labels_intersections.append("original")
        label_types.append(original_label)
        for label in intersection_label:
            inputs.append([model.entity_token, label])
            lengths.append(len(inputs[-1]))
            labels_intersections.append("intersection")
            label_types.append(original_label)

    embeddings = []
    for input, length in zip(inputs, lengths):
        out = model.token_rep_layer([input], torch.tensor([length]))
        entity_rep = out["embeddings"][0, 0::2]
        entity_rep = model.prompt_rep_layer(entity_rep)
        assert entity_rep.shape == (1, 768)
        embeddings.append(entity_rep[0].detach().cpu().numpy())

    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(np.stack(embeddings))
    scaler = MinMaxScaler()
    scaled_tsne = scaler.fit_transform(embeddings_tsne)
    plot_labels = labels_synonyms + labels_intersections
    df = pd.DataFrame.from_dict(
        {
            "x": scaled_tsne[:, 0],
            "y": scaled_tsne[:, 1],
            "granularity": plot_labels,
            "label": label_types,
            "dataset": len(label_types) * [dataset],
        }
    )

    df["dataset"] = df["dataset"].str.split("_").str[0].str.capitalize()

    return df


def plot(df):
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.5)
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        style="granularity",
        s=200,
        col="dataset",
        col_wrap=2,
        col_order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
        facet_kws={"legend_out": False},
    )

    handles, labels = g.axes[0].get_legend_handles_labels()
    handles = handles[1:]
    labels = labels[1:]
    idx = labels.index("granularity")
    labels[idx] = ""
    g.axes[0].legend_.remove()
    g.fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        frameon=False,
    )

    g.tight_layout()
    g.savefig("embedding_space.png", bbox_inches="tight")


if __name__ == "__main__":
    configs = {
        "pilener_train": ["2024-03-12_14-40-13/model_30000"],
        "ontonotes": ["2024-03-12_17-05-12/model_30000"],
        "litset": ["2024-03-13_16-26-19/model_30000"],
        "fewnerd": ["2024-03-13_09-53-49/model_30000"],
    }
    plot_data = pd.DataFrame()
    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-small/{dataset}/{model_path}"
            )

            model = load_model(model_ckpt)
            device = next(model.parameters()).device
            model.to(device)
            model.eval()

            df = analyze_embedding_space(model, dataset)
            plot_data = pd.concat([plot_data, df])

    plot(plot_data)
