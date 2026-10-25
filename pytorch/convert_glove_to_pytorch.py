#!/usr/bin/env python
"""Convert GloVe text vectors (default: 2024 Wikipedia + Gigaword 5, 50d) into a
PyTorch weight file usable by ``glove_embedding.GloVeEmbedding``.

The output ``.pt`` file contains a dict with:
    weight: FloatTensor [vocab_size + 2, dim]  (row 0 = <pad>, row 1 = <unk>)
    vocab:  list[str] of length vocab_size + 2
    dim:    int

Usage:
    python convert_glove_to_pytorch.py --download --output glove.2024.wikigiga.50d.pt
    python convert_glove_to_pytorch.py --input glove.2024.wikigiga.50d.txt --output out.pt
"""
import argparse
import io
import os
import urllib.request
import zipfile

import torch

URL = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def download(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(zip_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_names:
            raise ValueError(f"No .txt file found in {zip_path}")
        zf.extract(txt_names[0], dest_dir)
        return os.path.join(dest_dir, txt_names[0])


def load_glove_txt(path, dim=None):
    vocab, vectors, seen = [], [], set()
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if line_no == 1 and len(parts) == 2:
                try:
                    header_vocab_size, header_dim = map(int, parts)
                except ValueError:
                    pass
                else:
                    if header_vocab_size >= 0 and header_dim > 0:
                        if dim is not None and dim != header_dim:
                            raise ValueError(
                                f"Header dimension {header_dim} does not match requested dimension {dim}")
                        dim = header_dim
                        continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) != dim + 1:
                print(f"Skipping malformed line {line_no}")
                continue
            word = parts[0]
            if word in seen or word in (PAD_TOKEN, UNK_TOKEN):
                continue
            seen.add(word)
            vocab.append(word)
            vectors.append([float(x) for x in parts[1:]])
    return vocab, torch.tensor(vectors, dtype=torch.float32), dim


def convert(input_path, output_path, dim=None):
    words, vectors, dim = load_glove_txt(input_path, dim)
    pad = torch.zeros(1, dim)
    unk = vectors.mean(dim=0, keepdim=True)
    weight = torch.cat([pad, unk, vectors], dim=0)
    vocab = [PAD_TOKEN, UNK_TOKEN] + words
    torch.save({"weight": weight, "vocab": vocab, "dim": dim}, output_path)
    print(f"Saved {weight.shape[0]} x {dim} embedding to {output_path}")
    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="Path to GloVe .txt file")
    p.add_argument("--download", action="store_true",
                   help=f"Download vectors from {URL}")
    p.add_argument("--data-dir", default="data", help="Download directory")
    p.add_argument("--output", default="glove.2024.wikigiga.50d.pt")
    p.add_argument("--dim", type=int, default=None,
                   help="Vector dimension (inferred if omitted)")
    args = p.parse_args()
    if args.input:
        path = args.input
    elif args.download:
        path = download(URL, args.data_dir)
    else:
        p.error("provide --input or --download")
    convert(path, args.output, args.dim)


if __name__ == "__main__":
    main()
