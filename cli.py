import argparse
import json
import os
import logging

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from dataset import MultiModalDataset, MultiModalModule, MultiModalTestDataset
from model import MultiModalClassifier
from utils import TrainingConfig, save_states, load_encoders

seed_everything(42)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def fit(args: argparse.Namespace):
    options = json.loads(args.options)
    
    config = TrainingConfig(**options)

    dataset = MultiModalDataset(
        dataset_uri=args.dataset_uri,
        text_column=args.text_column,
        label_column=args.label_column,
        num_cols=args.num_columns,
        cat_cols=args.cat_columns
    )

    logging.info("Dataset prepared")

    data = MultiModalModule(
        dataset=dataset,
        num_cols=args.num_columns,
        cat_cols=args.cat_columns,
        config=config
    )

    model = MultiModalClassifier(dataset.label_names, config)

    checkpoint_cb = ModelCheckpoint()
    early_cb = EarlyStopping(monitor="val_loss")
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[early_cb, checkpoint_cb]
    )

    logging.info("Training is starting")

    trainer.fit(model, data)

    logging.info("Finished training")

    model = MultiModalClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
    metrics = trainer.test(model, data)[0]
    logging.info(metrics)

    if not os.path.exists(args.save_state_dir):
        os.makedirs(args.save_state_dir)

    save_states(model, dataset, args.save_state_dir)

    logging.info(f"best checkpoint path: {checkpoint_cb.best_model_path}")

    return checkpoint_cb.best_model_path


def predict(args: argparse.Namespace):
    model = MultiModalClassifier.load_from_checkpoint(args.model_checkpoint).eval()
    num_encoder, cat_encoders = load_encoders(args.save_state_dir)

    logging.info("Model is initialized")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dataset = MultiModalTestDataset(
        dataset_uri=args.dataset_uri,
        text_column=args.text_column,
        num_cols=args.num_columns,
        cat_cols=args.cat_columns,
        num_encoder=num_encoder,
        cat_encoders=cat_encoders
    )

    logging.info("Dataset prepared")

    with open(args.output_file, "a") as fout:
        for batch in DataLoader(dataset, batch_size=args.batch_size):
            pred = model(*batch)
            pred = torch.argmax(pred, dim=1).data.tolist()

            for p in pred:
                fout.write(f"{int(p)}\n")

    logging.info("Done predicting")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("-d", "--dataset-uri", type=str, help="path to data", required=True)
    fit_parser.add_argument("-t", "--text-column", type=str, help="name of the text column", required=True)
    fit_parser.add_argument("-l", "--label-column", type=str, help="name of the label column", required=True)
    fit_parser.add_argument("-c", "--cat-columns", type=str, nargs="+",
                            help="names of categorical columns", required=True)
    fit_parser.add_argument("-n", "--num-columns", type=str, nargs="+",
                            help="names of numerical columns", required=True)
    fit_parser.add_argument("-s", "--save-state-dir", type=str, help="path to save states to", required=True)
    fit_parser.add_argument("-o", "--options", default='{}', type=str,
                            help="training options as a JSON dict")
    fit_parser.set_defaults(callback=fit)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("-d", "--dataset-uri", type=str, help="path to data", required=True)
    predict_parser.add_argument("-t", "--text-column", type=str, help="name of the text column", required=True)
    predict_parser.add_argument("-c", "--cat-columns", type=str, nargs="+",
                            help="names of categorical columns", required=True)
    predict_parser.add_argument("-n", "--num-columns", type=str, nargs="+",
                            help="names of numerical columns", required=True)
    predict_parser.add_argument("-s", "--save-state-dir", type=str, help="path to the saved model", required=True)
    predict_parser.add_argument("-m", "--model-checkpoint", type=str, help="path to the saved checkpoint", required=True)
    predict_parser.add_argument("-o", "--output-file", type=str, help="output file name", required=True)
    predict_parser.add_argument("-b", "--batch-size", type=int, default=8, required=False)
    predict_parser.set_defaults(callback=predict)

    args = parser.parse_args()
    args.callback(args)