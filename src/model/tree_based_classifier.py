import shap
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model.base_model import BaseModel
from dataloader.dataset import TabularDataset
from evaluation.evaluation import Evaluation
from util.helpers import python_to_json, store_obj_to_disk


class TreeBasedClassifier(BaseModel):
    """
    Tree-based classification (Random Forest, GradientBoosting) for tabular data
    """
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.model = None

        # specificity is the recall of the negative class
        specificity = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, pos_label=0)
        self.binary_metrics = {
            'accuracy': metrics.accuracy_score,
            'f1': metrics.f1_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'specificity': specificity,
            'confusion_matrix': metrics.confusion_matrix
        }
        self.continous_metrics = {
            'roc_auc': metrics.roc_auc_score,
            'roc_curve': metrics.roc_curve
        }

        ## Some configuration

        # Use cross validation if cv_splits > 1. If cv_splits == 1, use predefined train and test sets
        try:
            self.cv_splits = self.config.cv_splits
        except AttributeError:
            self.cv_splits = 10

        # whether or not to store the model to disk for future analysis
        try:
            self.store_model = self.config.store_model
        except AttributeError:
            self.store_model = False

        try:
            self.n_estimators = self.config.config_model.n_estimators
        except (AttributeError, KeyError):
            self.n_estimators = 500

        # Learning rate, for gradient boosting
        try:
            self.learning_rate = self.config.config_model.learning_rate
        except (AttributeError, KeyError):
            self.learning_rate = 0.1

        print(f"Using store_model {self.store_model}, cv_splits {self.cv_splits}, n_estimators {self.n_estimators}, "
              f"learning_rate {self.learning_rate}")


    def _save_model(self, split_idx=None):
        # store model to disk so we can later explore it
        # only if store_model = True and for the first split
        if self.store_model and (split_idx == 1 or split_idx is None):
            raise NotImplementedError()

    def set_train(self, dataset: TabularDataset):
        self._train = dataset

    def set_test(self, dataset: TabularDataset):
        self._test = dataset

    def prepare_data(self):
        pass

    def train(self, train_set=None, split_idx=None):
        """
        One training using train_set, if given, or self._train otherwise

        :param train_set: train set of this CV split
        :param split_idx: CV split Number
        :return:
        """
        assert train_set is not None

        if self.name == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        elif self.name == 'GradientBoosting':
            self.model = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid model for tree-based classifier, can only be RandomForest or GradientBoosting")

        # Print some identifiers of items in sets
        # This is to be able to manually check that the splits are identical over different runs
        def subset_identifer(data_set):
            # get the sum of the tensor elements of the first and last sample in each split
            sums = [np.sum(row) for row in data_set.data.values]
            return [sums[0], sums[-1]]
        print(f"Train set: {subset_identifer(train_set)}...")

        if split_idx == 0 or split_idx is None:
            print("Model used:")
            print(self.model)

        print(f"Start training", f"split {split_idx}..." if split_idx is not None else "")

        self.model.fit(train_set.data, train_set.labels)


    def test(self, test_set=None, split_idx=None):
        """
        Test self.model on test_set, if given (for cross-validation), or self._test, if not
        """
        assert test_set is not None
        assert self.model is not None
        print("Start testing", f"split {split_idx}..." if split_idx is not None else "")

        predictions = self.model.predict_proba(test_set.data)[:,1]

        computed_metrics_binary = {name: self.binary_metrics[name](test_set.labels, np.round(predictions).astype(int))
                                   for name in self.binary_metrics}
        computed_metrics_continuous = {name: self.continous_metrics[name](test_set.labels, predictions)
                                       for name in self.continous_metrics}
        computed_metrics = {**computed_metrics_binary, **computed_metrics_continuous}
        computed_metrics['predictions'] = predictions
        computed_metrics['labels'] = test_set.labels

        # sklearn feature importance
        # for both RandomForestClassifier and GradientBoostingClassifier, this is
        # "The impurity-based feature importances, (...) also known as the Gini importance."
        computed_metrics['gini_feature_importance'] = self.model.feature_importances_

        # shap value explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(test_set.data)
        if isinstance(shap_values, list):
            # There are two outputs, for each output class, happens for RandomForest
            assert len(shap_values) == 2, f"Expected one shap_value output per class, but it's {len(shap_values)}"
            shap_values = shap_values[1]
        else:
            # Only one output, for GradientBoosting
            pass
        assert isinstance(shap_values, np.ndarray), "Shap_value output should be ndarray"
        computed_metrics['shap_values'] = shap_values

        return computed_metrics

    def train_test(self):
        test_metrics = {}
        all_predictions, all_labels = [], []  # for ROC curves and other further analysis
        all_sample_names = [] # for further analysis
        all_test_data, all_shap_values = [], []  # for summary plots over all splits
        all_gini_feature_importances = []

        complete_dataset = self._train.concatenate(self._test)

        # store dataset to disk, to do analysis on features
        complete_dataset.store_to_disk(os.path.join(self.run_parameters.results_dir, "dataset"))

        if self.cv_splits > 1:
            # We use cross validation, i.e. we first combine the train and test set and then use the pre-defined
            # (deterministic) mapping to split it
            complete_dataset.load_cv_split_assignment(self.cv_splits)

            for split_idx in range(self.cv_splits):
                test_indices = np.where(complete_dataset.cv_test_splits == split_idx)[0]
                train_indices = np.where(complete_dataset.cv_test_splits != split_idx)[0]
                train_set = complete_dataset.subset_from_indices(train_indices)
                test_set = complete_dataset.subset_from_indices(test_indices)

                self.train(train_set, split_idx)
                test_results = self.test(test_set, split_idx=split_idx)
                test_metrics[f'split_{split_idx}'] = test_results
                all_predictions.append(test_results['predictions'])
                all_labels.append(test_results['labels'])
                all_sample_names.append(test_set.sample_names)
                all_shap_values.append(test_results['shap_values'])
                all_test_data.append(test_set.data)
                all_gini_feature_importances.append(test_results['gini_feature_importance'])

                self._save_model(split_idx)

            print("CV test metrics:", test_metrics)

            all_predictions_flat = np.concatenate(all_predictions)
            all_labels_flat = np.concatenate(all_labels)
            computed_metrics_binary = {name: self.binary_metrics[name](all_labels_flat, np.round(all_predictions_flat).astype(int))
                for name in self.binary_metrics}
            computed_metrics_continuous = {name: self.continous_metrics[name](all_labels_flat, all_predictions_flat)
                                           for name in self.continous_metrics}
            computed_metrics = {**computed_metrics_binary, **computed_metrics_continuous}
            print("CV test metrics aggregated", computed_metrics)

        else:
            # No cross validation, use existing train / test split
            train_set = self._train
            test_set = self._test
            self.train(train_set)
            test_metrics = self.test(test_set)
            all_predictions.append(test_metrics['predictions'])
            all_labels.append(test_metrics['labels'])
            all_sample_names.append(test_set.sample_names)
            all_shap_values.append(test_metrics['shap_values'])
            all_test_data.append(test_set.data)
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
        # plot_roc(os.path.join(self.run_parameters.results_dir, "roc.png"), all_predictions, all_labels)
        # plot_roc_cv(os.path.join(self.run_parameters.results_dir, "roc_cv.png"), all_predictions, all_labels)

        # write gini feature importance to file
        self._write_gini_feature_importance(list(complete_dataset.data.columns), all_gini_feature_importances)

        # write feature correlation matrix to file
        self._write_correlation_matrix(complete_dataset.data)

        # write feature distributions per file to disk
        self._write_feature_distribution(complete_dataset)

        # write shap values to file
        all_shap_values_flat = np.concatenate(all_shap_values)
        all_test_data_flat = pd.concat(all_test_data)
        shap.summary_plot(all_shap_values_flat, all_test_data_flat, show=False)
        plt.savefig(os.path.join(self.run_parameters.results_dir, "shap.png"))
        store_obj_to_disk("shap_values", all_shap_values_flat, self.run_parameters.results_dir)

        print("Done.")

    def _write_gini_feature_importance(self, features, all_gini_feature_importances):
        features = np.array(features)
        importances = np.mean(all_gini_feature_importances, axis=0)  # mean over splits
        indices = np.argsort(importances)

        with open(os.path.join(self.run_parameters.results_dir, "gini_feature_importance.txt"), "w") as file:
            file.write(python_to_json(list(zip(features, importances))))

        plt.figure(figsize=(6, len(features)/4))
        plt.title('Gini feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance (gini)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "gini_feature_importance.png"))
        plt.close()

    def _write_correlation_matrix(self, data: pd.DataFrame):
        n_features = data.shape[1]
        f = plt.figure(figsize=(n_features*.3+4, n_features*.3))
        correlation_matrix = data.corr()
        plt.matshow(correlation_matrix, fignum=f.number)
        plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14,
                   rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_correlation.png"))
        plt.close()

        with open(os.path.join(self.run_parameters.results_dir, "feature_correlation.txt"), "w") as file:
            file.write(python_to_json(correlation_matrix))

    def _store_dataset(self, complete_dataset):
        complete_dataset.data.to_pickle("dataset_features.pkl")
        complete_dataset.save("dataset_labels")

    def _write_feature_distribution(self, dataset):
        def plot_distribution(features_dementia, features_control, all_features):
            feature_names = all_features.columns
            positions_dementia = [i * 3 for i in range(len(feature_names))]
            positions_control = [i * 3 + 1 for i in range(len(feature_names))]
            tick_positions = [i * 3 + 0.5 for i in range(len(feature_names))]

            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            # normalized values
            dementia_normalized = (features_dementia - mean) / std
            control_normalized = (features_control - mean) / std

            # diff
            diff = dementia_normalized.mean() - control_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            # sorted
            dementia_normalized = dementia_normalized[sorted_features]
            control_normalized = control_normalized[sorted_features]

            fig, ax2 = plt.subplots(1, 1, figsize=(len(feature_names)/4, 6))
            fig.suptitle(f'Distribution of normalized feature values Dementia vs. Control')
            ax2.boxplot(dementia_normalized.values, sym="", positions=positions_dementia,
                                         showmeans=True, meanline=True,
                                         patch_artist=True, boxprops=dict(facecolor="#ffa8a8"))
            ax2.boxplot(control_normalized.values, sym="", positions=positions_control,
                                        showmeans=True, meanline=True,
                                        patch_artist=True, boxprops=dict(facecolor="#a8d9ff"))
            ax2.set_xticks(tick_positions, sorted_features, rotation=45, ha='right', rotation_mode='anchor')
            ax2.set_ylabel("Normalized value of feature")
            ax2.axhline(0, linestyle=":", color="k", alpha=0.5)

            blue_patch = mpatches.Patch(color='#ffa8a8', label='Dementia')
            green_patch = mpatches.Patch(color='#a8d9ff', label='Control')
            fig.legend(handles=[blue_patch, green_patch])

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution.png"))
            plt.close()

        def plot_individual_distribution(features_dementia, features_control, all_features, figsize_individual):
            feature_names = all_features.columns

            n_cols = int(np.ceil(np.sqrt(len(feature_names))))
            n_rows = int(np.ceil(len(feature_names) / n_cols))
            print("n_rows, n_cols", n_rows, n_cols)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * figsize_individual[0], n_rows * figsize_individual[1]))

            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]

            # mean and std, normalize values, calculate difference, sort features accordingly
            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            dementia_normalized = (features_dementia - mean) / std
            control_normalized = (features_control - mean) / std

            diff = dementia_normalized.mean() - control_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            features_dementia = features_dementia[sorted_features]
            features_control = features_control[sorted_features]

            for feature, ax in zip(sorted_features, [col for row in axes for col in row]):
                ax.boxplot(features_dementia[feature], sym="", positions=[0],
                           showmeans=True, meanline=True)
                ax.boxplot(features_control[feature], sym="", positions=[1],
                           showmeans=True, meanline=True)
                ax.set_xticks([0, 1], ['Dementia', 'Control'], rotation=45, ha='right', rotation_mode='anchor')
                ax.set_ylabel("Feature Value")
                ax.set_title(feature, fontsize=10)
                ax.axhline(mean[feature], linestyle=":", color="k", alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution_individual.png"))
            plt.close()

        df = dataset.data
        df['label'] = dataset.labels
        all_features = df.drop(columns=['label'])
        features_dementia = df.query("label == 1").drop(columns=['label'])
        features_control = df.query("label == 0").drop(columns=['label'])

        plot_distribution(features_dementia, features_control, all_features)
        plot_individual_distribution(features_dementia, features_control, all_features, figsize_individual=(2, 2))


class RandomForest(TreeBasedClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__("RandomForest", *args, **kwargs)

class GradientBoosting(TreeBasedClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__("GradientBoosting", *args, **kwargs)