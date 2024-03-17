import torch
from torch import nn
import pytorch_lightning as pl
from utils import TrainingConfig
from transformers import AutoTokenizer, AutoModel
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


class MultiModalClassifier(pl.LightningModule):

    def __init__(
            self,
            label_names: list,
            config: TrainingConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.label_names = label_names

        num_labels = len(label_names)
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
        self.text_enc = AutoModel.from_pretrained(config.hf_model_name)
        self.head = nn.Sequential(
            nn.LazyLinear(config.fc_hidden_state),
            nn.LeakyReLU(),
            nn.LazyLinear(num_labels)
        )
        if config.freeze_text_enc:
            for param in self.text_enc.base_model.parameters():
                param.requires_grad = False
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_labels, average="macro"),
                "precision": Precision(task="multiclass", num_classes=num_labels, average="macro"),
                "recall": Recall(task="multiclass", num_classes=num_labels, average="macro"),
                "f1": F1Score(task="multiclass", num_classes=num_labels, average="macro")
            }
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, texts, meta):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.token_length
        )
        meta = meta.to(self.device)
        tokens = tokens.to(self.device)
        x = self.text_enc(**tokens)
        x = self._mean_pooling(x, tokens['attention_mask'])
        x = torch.concat([x, meta], dim=-1)
        x = self.head(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idxs):
        return self._shared_eval_step(batch, batch_idxs, "train")

    def validation_step(self, batch, batch_idxs):
        self._shared_eval_step(batch, batch_idxs, "val")

    def test_step(self, batch, batch_idxs):
        self._shared_eval_step(batch, batch_idxs, "test")

    def log_metrics(self, yhat, y, stage_name):
        for (metric_name, metric) in self.metrics.items():
            val = metric(yhat, y)
            self.log(f"{stage_name}_{metric_name}", val)

    def _shared_eval_step(self, batch, batch_idx, step_name):
        texts, meta, y = batch
        yhat = self(texts, meta)
        loss = self.loss_fn(yhat, y)
        self.log(f"{step_name}_loss", loss)
        self.log_metrics(yhat, y, step_name)
        return loss
