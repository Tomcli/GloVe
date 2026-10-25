"""PyTorch embedding model backed by converted GloVe vectors.

Example:
    from glove_embedding import GloVeEmbedding
    model = GloVeEmbedding.from_pretrained("glove.2024.wikigiga.50d.pt")
    vecs, mask = model(["Hello world!", "GloVe vectors"])  # [2, T, 50], [2, T]
    sent = model.encode(["Hello world!"])                  # [1, 50] mean-pooled
"""
import re

import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
# Words (with internal apostrophes/hyphens/periods), numbers, or single punctuation.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\-.][^\W_]+)*|[^\s\w]|_")


class GloVeTokenizer:
    """Lowercasing regex tokenizer matching the uncased 2024 GloVe vocab."""

    def __init__(self, vocab):
        self.itos = list(vocab)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def tokenize(self, text):
        return _TOKEN_RE.findall(text.lower())

    def encode(self, text):
        return [self.stoi.get(t, self.unk_id) for t in self.tokenize(text)]

    def batch_encode(self, texts):
        ids = [self.encode(t) for t in texts]
        max_len = max((len(x) for x in ids), default=0)
        input_ids = torch.full((len(ids), max_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(ids):
            if seq:
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return input_ids, input_ids != self.pad_id


class GloVeEmbedding(nn.Module):
    def __init__(self, weight, vocab, freeze=True):
        super().__init__()
        self.tokenizer = GloVeTokenizer(vocab)
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze, padding_idx=self.tokenizer.pad_id)

    @classmethod
    def from_pretrained(cls, path, freeze=True, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        return cls(ckpt["weight"], ckpt["vocab"], freeze=freeze)

    @property
    def dim(self):
        return self.embedding.embedding_dim

    def forward(self, texts):
        """texts: str or list[str] -> (embeddings [B, T, D], mask [B, T])."""
        if isinstance(texts, str):
            texts = [texts]
        input_ids, mask = self.tokenizer.batch_encode(texts)
        device = self.embedding.weight.device
        return self.embedding(input_ids.to(device)), mask.to(device)

    def encode(self, texts):
        """Mean-pooled sentence embeddings [B, D] (zeros for empty input)."""
        emb, mask = self(texts)
        m = mask.unsqueeze(-1).to(emb.dtype)
        return (emb * m).sum(1) / m.sum(1).clamp(min=1)

    def lookup(self, word):
        idx = self.tokenizer.stoi.get(word.lower(), self.tokenizer.unk_id)
        return self.embedding.weight[idx]
