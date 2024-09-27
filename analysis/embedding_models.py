import io
import os
import re
from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class LabelEmbeddingModel:
    def __init__(self):
        pass

    @abstractmethod
    def embed(self, batch: List[str]) -> Dict[str, np.array]:
        pass


class FastTextModel(LabelEmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str,
        data_dir: str = "/home/ec2-user/paper_data/embeddings",
    ):
        super().__init__()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        words, embeddings = self.load_vectors(model_name_or_path, data_dir)
        self.words = words
        self.embeddings = embeddings

    def load_vectors(self, fname: str, data_dir: str):
        fin = io.open(
            os.path.join(data_dir, fname),
            "r",
            encoding="utf-8",
            newline="\n",
            errors="ignore",
        )
        n, d = map(int, fin.readline().split())
        words = []
        embeddings = []
        for line in tqdm(fin.readlines(), desc="Loading FastText"):
            tokens = line.rstrip().split(" ")
            words.append(tokens[0])
            embeddings.append(torch.tensor(list(map(float, tokens[1:]))))

        words = {w: i for i, w in enumerate(words)}
        words[self.unk_token] = len(words)
        words[self.pad_token] = len(words)

        embeddings = torch.stack(embeddings)
        unk_embedding = torch.mean(embeddings, dim=0)
        padding_embedding = torch.zeros(1, embeddings.size(1))
        embeddings = torch.cat(
            [embeddings, unk_embedding.unsqueeze(0), padding_embedding], dim=0
        )
        embeddings = torch.nn.Embedding.from_pretrained(embeddings).to(get_device())
        return words, embeddings

    def embed(self, batch: List[str]) -> Dict[str, np.array]:
        nested_batch = [re.split(r"[-/_ ]", label.lower()) for label in batch]
        max_length = max(len(inner_list) for inner_list in nested_batch)

        input_ids = torch.LongTensor(
            [
                [
                    self.words.get(label, self.words.get(self.unk_token))
                    for label in labels
                ]
                + [self.words.get(self.pad_token)] * (max_length - len(labels))
                for labels in nested_batch
            ]
        ).to(get_device())

        mask = input_ids != self.words.get(self.pad_token)

        embeddings = torch.sum(self.embeddings(input_ids), dim=1) / mask.sum(
            dim=1
        ).unsqueeze(1)
        return dict(zip(batch, embeddings.cpu().numpy()))


class GloveModel(LabelEmbeddingModel):
    def __init__(
        self, glove_file: str, glove_dir: str = "/home/ec2-user/paper_data/embeddings"
    ):
        super().__init__()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        words, embeddings = self.load_glove(glove_file, glove_dir)
        self.words = words
        self.embeddings = embeddings

    def load_glove(
        self, glove_file: str, glove_dir: str = "/home/ec2-user/paper_data/embeddings"
    ) -> Tuple[Dict[str, int], torch.nn.Embedding]:
        """
        Load GloVe model from a text file.

        Args:
            glove_file (str): Path to the GloVe text file.

        Returns:
            dict: A dictionary mapping words to their GloVe vector representations.
        """
        word_embedding_pairs = []
        with open(os.path.join(glove_dir, glove_file), "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Loading GloVe"):
                parts = line.split(" ")
                word = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]])
                word_embedding_pairs.append((word, vector))

        words, embeddings = zip(*word_embedding_pairs)
        words = {w: i for i, w in enumerate(words)}
        words[self.unk_token] = len(words)
        words[self.pad_token] = len(words)

        embeddings = torch.stack(embeddings)
        unk_embedding = torch.mean(embeddings, dim=0)
        padding_embedding = torch.zeros(1, embeddings.size(1))
        embeddings = torch.cat(
            [embeddings, unk_embedding.unsqueeze(0), padding_embedding], dim=0
        )
        embeddings = torch.nn.Embedding.from_pretrained(embeddings).to(get_device())
        return words, embeddings

    def embed(self, batch: List[str]) -> Dict[str, np.array]:
        nested_batch = [re.split(r"[-/_ ]", label.lower()) for label in batch]
        max_length = max(len(inner_list) for inner_list in nested_batch)

        input_ids = torch.LongTensor(
            [
                [
                    self.words.get(label, self.words.get(self.unk_token))
                    for label in labels
                ]
                + [self.words.get(self.pad_token)] * (max_length - len(labels))
                for labels in nested_batch
            ]
        ).to(get_device())

        mask = input_ids != self.words.get(self.pad_token)

        embeddings = torch.sum(self.embeddings(input_ids), dim=1) / mask.sum(
            dim=1
        ).unsqueeze(1)
        return dict(zip(batch, embeddings.cpu().numpy()))


class TransformerModel(LabelEmbeddingModel):
    def __init__(self, model_name_or_path: str, pooling: str = "mean"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path).to(get_device())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pooling = pooling

    def embed(self, batch: List[str]) -> Dict[str, np.array]:
        inputs = self.tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(get_device())
        outputs = self.model(**inputs)
        if self.pooling == "mean":
            return dict(zip(batch, outputs.last_hidden_state.mean(dim=1).cpu().numpy()))
        else:
            return dict(zip(batch, outputs.last_hidden_state[:, 0, :].cpu().numpy()))


class SentenceTransformerModel(LabelEmbeddingModel):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path).to(get_device())

    def embed(self, batch: List[str]) -> Dict[str, np.array]:
        embedding = self.model.encode(batch, convert_to_tensor=True)
        return dict(zip(batch, embedding.cpu().numpy()))


def get_device():
    """Whether to use GPU or CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name_or_path: str) -> torch.nn.Embedding:
    if "glove" in model_name_or_path:
        model = GloveModel(model_name_or_path)
    elif "sentence-transformers" in model_name_or_path:
        model = SentenceTransformerModel(model_name_or_path)
    elif "wiki-news" in model_name_or_path or "crawl" in model_name_or_path:
        model = FastTextModel(model_name_or_path)
    else:
        model = TransformerModel(model_name_or_path)
    return model
