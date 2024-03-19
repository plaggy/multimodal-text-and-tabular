Usage (see`cli.py`):

```
python cli.py fit ...
python cli.py predict ...
```

- use `TrainingConfig` in `utils.py` to set (hyper)parameters
- `dataset.py` - dataset wrapper implementation, the expected data format is described in `MultiModalDataset`. `MultiModalTestDataset` may be useful for batch inference, not really needed for online inference
- `MultiModalClassifier` in `model.py` implements the architecture itself. Metrics and a loss function are also defined there
- `cli.py` implements a training loop and offline prediction. Note that `LabelEncoder` and `MinMaxScaler` used for tabular data processing are also pickled to be reused on inference. 



Not intended to be as a go-to solution, but rather a comprehensive example of a custom multimodal model that handles both text and tabular data simultaneously.
In Pytorch-Lightning
