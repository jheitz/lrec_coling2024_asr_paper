import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, BertTokenizer
from transformers import AutoModelForSequenceClassification, BertModel
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torchmetrics
import numpy as np

from model.base_model import BaseModel
from dataloader.dataset import TextDataset
from evaluation.evaluation import Evaluation
from util.helpers import python_to_json, plot_roc, plot_roc_cv, plot_learning_rates

class BERT(BaseModel):
    """
    BERT Dementia classification on text
    Originally based on https://huggingface.co/docs/transformers/tasks/sequence_classification
    """
    def __init__(self, *args, **kwargs):
        super().__init__("BERT", *args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=self.CONSTANTS.CACHE_DIR)
        self._train: TextDataset = None
        self._test: TextDataset = None

        self.id2label = {0: "CONTROL", 1: "AD"}
        self.label2id = {"CONTROL": 0, "AD": 1}
        # self._accuracy = evaluate.load("accuracy")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.batch_size = 8
        self.evaluation_test = Evaluation(self.device, metrics=(
            torchmetrics.Accuracy('binary'), torchmetrics.F1Score('binary'), torchmetrics.Precision('binary'),
            torchmetrics.Recall('binary'), torchmetrics.Specificity('binary'), torchmetrics.AUROC('binary'),
            torchmetrics.ConfusionMatrix('binary'), torchmetrics.ROC('binary')))
        self.evaluation_train = Evaluation(self.device, metrics=(
            torchmetrics.Accuracy('binary'), torchmetrics.F1Score('binary'), torchmetrics.Precision('binary'),
            torchmetrics.Recall('binary'), torchmetrics.Specificity('binary'), torchmetrics.AUROC('binary')))

        ## Some configuration

        # number of epochs to train
        try:
            self.num_epochs = self.config.num_epochs
        except AttributeError:
            self.num_epochs = 10

        # Whether to use a validation set for validation after each epoch.
        # If not, all of the train set is used for training
        try:
            self.use_val_set = self.config.use_val_set
        except AttributeError:
            self.use_val_set = True

        # Use cross validation if cv_splits > 1. If cv_splits == 1, use predefined train and test sets
        try:
            self.cv_splits = self.config.cv_splits
        except AttributeError:
            self.cv_splits = 10

        # learning rate
        try:
            self.learning_rate = self.config.learning_rate
        except AttributeError:
            self.learning_rate = 4e-6

        # whether or not to store the model to disk for future analysis
        try:
            self.store_model = self.config.store_model
        except AttributeError:
            self.store_model = False

        print(f"Using num_epochs {self.num_epochs}, use_val_set {self.use_val_set}, "
              f"cv_splits {self.cv_splits}, learning_rate {self.learning_rate}, lr_schedule {self.lr_schedule}")

    def _load_model(self):
        self.model: BertModel = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2, id2label=self.id2label, label2id=self.label2id)

    def _save_model(self, split_idx=None):
        # store model to disk so we can later explore it
        # only if store_model = True and for the first split
        if self.store_model and (split_idx == 1 or split_idx is None):
            dirname = "model" if split_idx is None else f"model_split{split_idx}"
            dirpath = os.path.join(self.run_parameters.results_dir, dirname)
            self.model.save_pretrained(dirpath)

    def set_train(self, dataset: TextDataset):
        self._train = dataset

    def set_test(self, dataset: TextDataset):
        self._test = dataset

    def prepare_data(self):
        # Tokenize data for BERT
        def preprocess_function(ds: TextDataset):
            ds.tokens = np.array(self.tokenizer(list(ds.data), max_length=512, truncation=True, padding='max_length').input_ids)
            return ds

        self._train = preprocess_function(self._train)
        if self._test:
            self._test = preprocess_function(self._test)

    def train(self, train_set=None, split_idx=None):
        """
        One training using train_set, if given, or self._train otherwise

        :param train_set: train set of this CV split
        :param split_idx: CV split Number
        :return:
        """
        assert train_set is not None

        self._load_model()

        # split the train set into train / val, if required
        if self.use_val_set:
            train_set, val_set = torch.utils.data.random_split(train_set, (0.8, 0.2), torch.Generator().manual_seed(42))
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        else:
            val_set, val_loader = None, None
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # Print some identifiers of items in sets
        # This is to be able to manually check that the splits are identical over different runs
        def subset_identifer(data_set):
            # get the sum of the tensor elements of the first three samples in each split
            return [torch.sum(input_ids) for input_ids, labels in list(data_set)][:3]
        if self.use_val_set:
            print(f"Train / val set: {subset_identifer(train_set)}... / {subset_identifer(val_set)}...")
        else:
            print(f"Train set: {subset_identifer(train_set)}...")

        print(f"Training for {self.num_epochs} epochs")

        num_training_steps = self.num_epochs * len(train_loader)
        optimizer_warmup, optimizer_main, learning_rate_scheduler = self._set_learning_rate_schedule()

        if split_idx == 0 or split_idx is None:
            print("Model used:")
            print(self.model)

        self.model.to(self.device)

        print(f"Start training", f"split {split_idx}..." if split_idx is not None else "")

        progress_bar = tqdm(range(num_training_steps))

        # Empty cache
        torch.cuda.empty_cache()

        # log learning rates in time
        learning_rate_log = pd.DataFrame(data={}, index=[pg.get('name','param') for pg in optimizer_main.param_groups])

        # set optimizer for warmup: only train classifier head, not other layers
        optimizer = optimizer_warmup

        for epoch in range(self.num_epochs):
            self.model.train()
            learning_rate_epoch = pd.DataFrame(data={epoch: [pg['lr'] for pg in optimizer.param_groups]},
                                               index=[pg.get('name','param') for pg in optimizer.param_groups])
            learning_rate_log = learning_rate_log.join(learning_rate_epoch, how="outer")

            for batch in train_loader:
                input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                progress_bar.update(1)

            # decay learning rate after each epoch
            learning_rate_scheduler.step()

            self.model.eval()

            # Evaluation metrics using the val set, if available
            if val_loader is not None:
                for batch in val_loader:
                    input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)

                    logits = outputs.logits  # 2-dim for each sample
                    probabilities = F.softmax(logits, dim=1)[:,1]  # just keep the probab of the 1 class
                    self.evaluation_train.add_batch(predictions=probabilities, labels=labels, loss=outputs.loss)
                    # print("Eval batch loss", outputs.loss)

                print(f"\nEvaluation metrics", f"split {split_idx}..." if split_idx is not None else "",
                      f": {python_to_json(self.evaluation_train.compute())}")

            # warmup done -> change to main optimizer
            if epoch + 1 >= 3:
                optimizer = optimizer_main

        print(f"Finished training", f"split {split_idx}..." if split_idx is not None else "")


    def test(self, test_set=None, split_idx=None):
        """
        Test self.model on test_set, if given (for cross-validation), or self._test, if not
        """
        assert test_set is not None
        print("Start testing", f"split {split_idx}..." if split_idx is not None else "")

        test_loader = DataLoader(test_set, batch_size=self.batch_size)
        for batch in test_loader:
            input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)

            logits = outputs.logits  # 2-dim for each sample
            probabilities = F.softmax(logits, dim=1)[:, 1]  # just keep the probab of the 1 class
            self.evaluation_test.add_batch(predictions=probabilities, labels=labels, loss=outputs.loss)
            # print("Test batch loss", outputs.loss)

        computed_metrics = self.evaluation_test.compute()
        print(f"Test metrics", f"split {split_idx}..." if split_idx is not None else "", ":",
              python_to_json(computed_metrics))

        return computed_metrics

    def train_test(self):
        test_metrics = {}
        all_predictions, all_labels = [], []  # for ROC curves and other further analysis
        all_sample_names = [] # for further analysis

        if self.cv_splits > 1:
            # We use cross validation, i.e. we first combine the train and test set and then use the pre-defined
            # (deterministic) mapping to split it
            dataset = self._train.concatenate(self._test)
            dataset.load_cv_split_assignment(self.cv_splits)

            for split_idx in range(self.cv_splits):
                test_indices = np.where(dataset.cv_test_splits == split_idx)[0]
                train_indices = np.where(dataset.cv_test_splits != split_idx)[0]
                train_set = dataset.subset_from_indices(train_indices)
                test_set = dataset.subset_from_indices(test_indices)

                self.train(train_set.asTorchDataset(), split_idx)
                test_results = self.test(test_set.asTorchDataset(), split_idx=split_idx)
                test_metrics[f'split_{split_idx}'] = test_results
                all_predictions.append(test_results['predictions'])
                all_labels.append(test_results['labels'])
                all_sample_names.append(test_set.sample_names)

                self._save_model(split_idx)

            print("CV test metrics:", test_metrics)
            print("CV test metrics aggregated", {})

        else:
            # No cross validation, use existing train / test split
            train_set = self._train
            test_set = self._test
            self.train(train_set.asTorchDataset())
            test_metrics = self.test(test_set.asTorchDataset())
            all_predictions.append(test_metrics['predictions'])
            all_labels.append(test_metrics['labels'])
            all_sample_names.append(test_set.sample_names)
            self._save_model()

            print("Test metrics:", test_metrics)


        # write metrics to file (as json)
        with open(os.path.join(self.run_parameters.results_dir, "metrics.txt"), "w") as file:
            file.write(python_to_json(test_metrics))

        # write test predictions, labels, sample_names to file
        with open(os.path.join(self.run_parameters.results_dir, "predictions.txt"), "w") as file:
            file.write(python_to_json(all_predictions))
        with open(os.path.join(self.run_parameters.results_dir, "labels.txt"), "w") as file:
            file.write(python_to_json(all_labels))
        with open(os.path.join(self.run_parameters.results_dir, "sample_names.txt"), "w") as file:
            file.write(python_to_json(all_sample_names))

        # write roc curve to results dir
        plot_roc(os.path.join(self.run_parameters.results_dir, "roc.png"), all_predictions, all_labels)
        plot_roc_cv(os.path.join(self.run_parameters.results_dir, "roc_cv.png"), all_predictions, all_labels)