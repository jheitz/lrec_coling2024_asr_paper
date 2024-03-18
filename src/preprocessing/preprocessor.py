from abc import abstractmethod
from dataloader.dataset import Dataset

class Preprocessor:
    name = "Generic preprocessor"

    def __init__(self, config, constants):
        self.config = config
        self.CONSTANTS = constants

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        pass