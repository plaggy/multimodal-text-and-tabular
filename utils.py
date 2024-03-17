import csv
import json
import os
import pickle
import torch

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    lr: float = 2e-5
    fc_hidden_state: int = 512
    max_epochs: int = 10
    token_length: int = 256
    freeze_text_enc: bool = False
    hf_model_name: str = "microsoft/deberta-v3-base"
    batch_size: int = 8
    splits: list | tuple = (0.8, 0.1, 0.1)


def csv_iterator(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for record in reader:
            yield record


def jsonlines_iterator(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def dataset_iterator(dataset_uri):
    fmt, path = dataset_uri.split(":")
    if fmt == "csv":
        return csv_iterator(path)
    if fmt == "jsonlines":
        return jsonlines_iterator(path)

    raise ValueError(f"format {fmt} is not recognized, expected csv or jsonlines")


def save_states(
        model,
        dataset,
        save_state_dir
):
    model.tokenizer.save_pretrained(os.path.join(save_state_dir, "tokenizer"))
    model.text_enc.save_pretrained(os.path.join(save_state_dir, "text_enc"))
    torch.save(model.head.state_dict(), os.path.join(save_state_dir, "head.pt"))

    with open(os.path.join(save_state_dir, "label_names.json"), "w") as f:
        json.dump(model.label_names, f)

    with open(os.path.join(save_state_dir, "num_encoder.pkl"), "wb") as f:
        pickle.dump(dataset.num_encoder, f)

    for i, enc in enumerate(dataset.cat_encoders):
        with open(os.path.join(save_state_dir, f"cat_encoder{i}.pkl"), "wb") as f:
            pickle.dump(enc, f)

    with open(os.path.join(save_state_dir, "config.json"), "w") as f:
        json.dump(model.config.dict(), f)


def load_encoders(save_state_dir):

    with open(os.path.join(save_state_dir, "num_encoder.pkl"), "rb") as f:
        num_encoder = pickle.load(f)

    cat_encoders = []
    for i, path in enumerate(sorted([path for path in os.listdir(save_state_dir) if path.startswith("cat_encoder")])):
        with open(os.path.join(save_state_dir, path), "rb") as f:
            cat_encoders.append(pickle.load(f))

    return num_encoder, cat_encoders