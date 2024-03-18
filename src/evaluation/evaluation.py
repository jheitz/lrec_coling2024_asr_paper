import evaluate
import torchmetrics.classification
from sklearn.metrics import confusion_matrix
from torch import Tensor
import torch
import torchmetrics

class Evaluation:
    def __init__(self, device, metrics=(torchmetrics.Accuracy('binary'), torchmetrics.F1Score('binary'),
                                        torchmetrics.Precision('binary'), torchmetrics.Recall('binary'),
                                        torchmetrics.Specificity('binary'), torchmetrics.AUROC('binary'),
                                        torchmetrics.ConfusionMatrix('binary'), torchmetrics.ROC('binary'))):
        """
        :param metrics: Torchmetrics metrics for binary classification
        """
        self.metrics = [m.to(device) for m in metrics]

        self.device = device

        # data collected in current batch
        self._initialize_current_timestep()

        # metrics collected over time steps
        self.history = []

    def _initialize_current_timestep(self):
        self.current_timestep = {
            "predictions": None,
            "labels": None,
            "summed_loss": 0
        }

    def add_batch(self, predictions, labels, loss=None):
        """
        Add a batch of predictions / labels, to compute the metrics after some batches
        :param predictions: predicted scores (probabiltiy of true class)
        :param labels: true labels
        :param loss: loss for batch, per item
        """
        assert isinstance(predictions, Tensor), type(predictions)
        assert isinstance(labels, Tensor), type(labels)
        assert len(predictions.shape) == 1  # should be 1-dimensional
        assert len(labels.shape) == 1  # should be 1-dimensional
        assert len(predictions) == len(labels)
        assert all([0 <= p <= 1 for p in predictions])
        assert all([0 <= l <= 1 for l in labels])
        assert loss is not None
        assert isinstance(loss, Tensor), type(loss)
        assert len(loss.shape) == 0
        if loss is not None:
            assert loss > 0

        if self.current_timestep['predictions'] is not None:
            self.current_timestep['predictions'] = torch.cat((self.current_timestep['predictions'], predictions))
            self.current_timestep['labels'] = torch.cat((self.current_timestep['labels'], labels))
        else:
            self.current_timestep['predictions'] = predictions
            self.current_timestep['labels'] = labels
        if loss is not None:
            # the loss is an average over all items in the batch. We multiply it by the size to calculate
            # the total average loss correctly
            self.current_timestep['summed_loss'] += loss * len(predictions)

    def compute(self, predictions=None, labels=None, loss=None):
        """
        Compute metrics given predictions / labels, if given. Otherwise consume the previously passed
        predictions and labels from self.current_timestep
        :param predictions: predicted scores
        :param labels: true labels
        :param loss: loss (for plotting)
        """
        assert (predictions is None and labels is None) or (predictions is not None and labels is not None)

        if predictions is None:
            n_items = len(self.current_timestep['predictions'])

            # call recursively with self.current_timestamp
            outputs = self.compute(predictions=self.current_timestep['predictions'],
                                   labels=self.current_timestep['labels'],
                                   loss=self.current_timestep['summed_loss'] / n_items)  # average loss per item

            # reset self.current_timestep, as this method consumes it
            self._initialize_current_timestep()
            return outputs

        computed_metrics = {str(m): m(predictions, labels) for m in self.metrics}

        output = {**computed_metrics, 'loss': loss, 'n_items': len(predictions), 'predictions': predictions,
                  'labels': labels}
        return output





