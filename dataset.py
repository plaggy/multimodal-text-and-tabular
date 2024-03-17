import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from typing import Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, random_split

from utils import dataset_iterator, TrainingConfig


class MultiModalDataset(torch.utils.data.Dataset):
    """
    Assumes the following data format:
    - a single text column
    - a single label column
    - a list of numerical columns
    - a list of categorical columns
    """
    def __init__(
            self,
            dataset_uri: str,
            text_column: str,
            label_column: str,
            num_cols: list[str],
            cat_cols: list[str],
    ):
        it = dataset_iterator(dataset_uri)
        self.texts = []
        raw_labels = []
        self.meta_num = []
        self.meta_cat = []
        for rec in it:
            self.texts.append(rec[text_column])
            raw_labels.append(rec[label_column])
            self.meta_num.append([float(rec[col]) for col in num_cols])
            self.meta_cat.append([str(rec[col]) for col in cat_cols])
        self.label_names = sorted(list(set(raw_labels)))
        labels_inverse = {lbl: i for i, lbl in enumerate(self.label_names)}
        self.labels = [labels_inverse[lbl] for lbl in raw_labels]
        self.num_encoder = None
        self.cat_encoders = None

    def __getitem__(self, i):
        cat_meta = torch.concat([
            F.one_hot(
                torch.tensor(enc.transform([val])[0]),
                num_classes=len(enc.classes_)
            )
            for val, enc in zip(self.meta_cat[i], self.cat_encoders)
        ])
        num_meta = torch.tensor(self.num_encoder.transform([self.meta_num[i]])[0])
        return (
            self.texts[i],
            torch.cat([cat_meta, num_meta]).float(),
            self.labels[i],
        )

    def __len__(self):
        return len(self.labels)


class MultiModalTestDataset(torch.utils.data.Dataset):
    """
    The format of data is the same as for the MultiModalDataset
    """
    def __init__(
            self,
            dataset_uri: str,
            text_column: str,
            num_cols: list[str],
            cat_cols: list[str],
            num_encoder: MinMaxScaler,
            cat_encoders: list[LabelEncoder]
    ):
        it = dataset_iterator(dataset_uri)
        self.texts = []
        self.meta_num = []
        self.meta_cat = []
        for rec in it:
            self.texts.append(rec[text_column])
            self.meta_num.append([float(rec[col]) for col in num_cols])
            self.meta_cat.append([str(rec[col]) for col in cat_cols])
        self.num_encoder = num_encoder
        self.cat_encoders = cat_encoders

    def __getitem__(self, i):
        cat_meta = torch.concat([
            F.one_hot(
                torch.tensor(enc.transform([val])[0]),
                num_classes=len(enc.classes_)
            )
            for val, enc in zip(self.meta_cat[i], self.cat_encoders)
        ])
        num_meta = torch.tensor(self.num_encoder.transform([self.meta_num[i]])[0])
        return (
            self.texts[i],
            torch.cat([cat_meta, num_meta]).float()
        )

    def __len__(self):
        return len(self.texts)


class MultiModalModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset,
            num_cols: list[str],
            cat_cols: list[str],
            config: TrainingConfig
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.splits = config.splits
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.setup()

    def setup(self, stage: Optional[str] = None):
        assert len(self.splits) == 3, "'splits' must have exactly 3 elements"
        train_len = int(len(self.dataset) * self.splits[0])
        val_len = int(len(self.dataset) * self.splits[1])
        test_len = len(self.dataset) - train_len - val_len
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [train_len, val_len, test_len]
        )
        train_idc = set(self.train_set.indices)
        cat_encoders = []
        for i in range(len(self.dataset.meta_cat[0])):
            cat_list = [item[i] for j, item in enumerate(self.dataset.meta_cat) if j in train_idc]
            cat_encoders.append(LabelEncoder().fit(cat_list))

        self.dataset.cat_encoders = cat_encoders
        self.dataset.num_encoder = MinMaxScaler().fit(self.dataset.meta_num)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)