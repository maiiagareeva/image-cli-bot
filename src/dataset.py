from datasets import load_dataset

class VLMDataset:
    def __init__(self,data):
        ds = load_dataset(data.dataset)
        self.train_ds = ds["train"]
        self.eval_ds=ds["val"]