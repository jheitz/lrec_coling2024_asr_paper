import os, shutil
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
import torch
import re
import hashlib
import pickle
import urllib.parse
from google.cloud.speech_v2.types.cloud_speech import AutoDetectDecodingConfig

def create_directory(path, empty_dir = False):
    """ Creates directory if it doesnt exist yet, optionally deleting all files in there """
    if not os.path.exists(path):
        os.makedirs(path)

    if empty_dir:
        shutil.rmtree(path)
        os.makedirs(path)

    try:
        os.chmod(path, 0o777)
    except PermissionError:
        pass


def implies(a, b):
    # returns the logical a => b, i.e. not(a) or b
    return not a or b


def if_and_only_if(a, b):
    # return the logical a ⇔ b (if and only if)
    return implies(a, b) and implies(b, a)


def python_to_json(d):
    # dump python data to json with some special handling
    # this is used to write to easily handleable text files
    def process(data):
        if isinstance(data, dict):
            return {key: process(data[key]) for key in data}  # recursive
        if isinstance(data, list):
            return [process(list_item) for list_item in data]  # recursive
        if isinstance(data, tuple):
            return tuple([process(list_item) for list_item in data])  # recursive
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return process(data.to_dict())
        elif isinstance(data, AutoDetectDecodingConfig):
            return str(data)
        else:
            return data

    return json.dumps(process(d))


def plot_roc_cv(file_path, predictions_torch, labels_torch):
    """
    ROC Curve for CV: One line per CV split, then average and std over splits
    """
    predictions_numpy = [p.cpu().numpy() for p in predictions_torch]
    labels_numpy = [l.cpu().numpy() for l in labels_torch]

    # calculate individual ROC curves
    all_fpr_numpy, all_tpr_numpy, all_auroc = [], [], []
    for labels, predictions in zip(labels_numpy, predictions_numpy):
        fpr, tpr, _ = sk_metrics.roc_curve(labels, predictions)
        auroc = sk_metrics.auc(fpr, tpr)
        all_fpr_numpy.append(fpr)
        all_tpr_numpy.append(tpr)
        all_auroc.append(auroc)

    # interpolated tprs of the individual splits and base fpr for average curve
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    # mean and std for average curve
    mean_auc = np.mean(all_auroc, axis=0)
    std_auc = np.std(all_auroc, axis=0)

    # plot individual curves
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for i, (fpr, tpr, auroc) in enumerate(zip(all_fpr_numpy, all_tpr_numpy, all_auroc)):
        plt.plot(fpr, tpr, label='Split {} (area = {:.2f})'.format(i, auroc), lw=1)

        # interpolate according to base_fpr -> this is to allow for averaging
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.plot(base_fpr, mean_tpr, label=r"Mean ROC (area = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=4)
    plt.fill_between(
        base_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(file_path)


def plot_roc(file_path, predictions_torch, labels_torch):
    """
    Write ROC curve to file.
    Each split will get one fine line, and an entire ROC curve is printed in bold
    :param file_path: Path to file to be written
    :param predictions_torch: list of tensors of predictions per split, or tensor of predictions
    :param labels_torch: list of tensors of labels per split, or tensor of predictions
    """

    if isinstance(predictions_torch, list):
        # This is from a CV experiment
        is_cv = True
        assert isinstance(labels_torch, list)
        assert len(predictions_torch) == len(labels_torch)
        assert all([isinstance(p, torch.Tensor) for p in predictions_torch])
        assert all([isinstance(l, torch.Tensor) for l in labels_torch])
    else:
        is_cv = False
        predictions_torch = [predictions_torch]
        labels_torch = [labels_torch]

    predictions_numpy = [p.cpu().numpy() for p in predictions_torch]
    labels_numpy = [l.cpu().numpy() for l in labels_torch]

    all_predictions_numpy = np.concatenate(predictions_numpy)
    all_labels_numpy = np.concatenate(labels_numpy)


    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # draw fine line for each split
    for i, (fpr, tpr, _) in enumerate([sk_metrics.roc_curve(labels, predictions) for labels, predictions in zip(labels_numpy, predictions_numpy)]):
        auroc = sk_metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='Split {} (area = {:.2f})'.format(i, auroc), lw=1)

    # draw thicker line for total results
    fpr, tpr, _ = sk_metrics.roc_curve(all_labels_numpy, all_predictions_numpy)
    auroc = sk_metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r"Overall ROC (area = %0.2f)" % auroc, lw=4)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(file_path)


def get_sample_name_from_path(path):
    """
    Takes the URL path of a input file (audio file, transcription) and returns the sample_name of the corresponding sample,
    depending on the dataset.
    E.g. '/home/ubuntu/methlab/Students/Jonathan/data/dementiabank_extracted/0extra/ADReSS-IS2020-data/test/transcription/S160.cha' --> S160
    """

    basename = os.path.basename(path)
    regex = r"^(.*)\.(cha|mp3|wav)$"
    parts = re.search(regex, basename)
    try:
        sample_name = parts.group(1)
    except:
        sample_name = ""
        print(f"Error getting sample_name from path {path}. Setting to empty string")

    return sample_name


def get_sample_names_from_paths(paths: np.ndarray):
    assert isinstance(paths, np.ndarray)
    assert paths.shape[0] > 1 and (len(paths.shape) == 1 or paths.shape[1] == 1), \
        f"paths should be a 2d column vector or 1d vector, but has dimensions {paths.shape}"

    vectorized_extract = np.vectorize(get_sample_name_from_path)
    return vectorized_extract(paths)



def hash_from_dict(config: dict, hash_len=None):
    """
    Create hexadecimal hash of length len from a dictionary d
    Can be used to create directory for storing temporary results etc.
    """
    hash = hashlib.sha1(bytes(pickle.dumps(config))).hexdigest()
    if hash_len is not None:
        assert 0 < hash_len < len(hash)
        hash = hash[:hash_len]
    return hash


def dataset_name_to_url_part(name: str):
    """
    Make database name good for part of a url (e.g. directory name)
    """
    return urllib.parse.quote(name.replace(" ", "_").replace("(", "").replace(")", ""))



def store_obj_to_disk(obj_name, obj, base_path):
    # store obj to file, depending on type
    if isinstance(obj, np.ndarray):
        file_path = f"{obj_name}.npy"
        obj_type = "numpy"
        with open(os.path.join(base_path, file_path), 'wb') as f:
            np.save(f, obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        file_path = f"{obj_name}.pkl"
        obj_type = "pandas"
        obj.to_pickle(os.path.join(base_path, file_path))
    elif isinstance(obj, str):
        file_path = f"{obj_name}.txt"
        obj_type = "text"
        with open(os.path.join(base_path, file_path), "w") as f:
            f.write(obj)
    else:
        file_path = f"{obj_name}.pkl"
        obj_type = "pickle"
        with open(os.path.join(base_path, file_path), "wb") as f:
            pickle.dump(obj, f)

    return file_path, obj_type

def get_obj_from_disk(file_path, obj_type, base_path):
    # get obj from file, which has previously been written to there using above function store_obj_to_disk
    if obj_type == "numpy":
        with open(os.path.join(base_path, file_path), 'rb') as f:
            obj = np.load(f)
    elif obj_type == "pandas":
        obj = pd.read_pickle(os.path.join(base_path, file_path))
    elif obj_type == "text":
        with open(os.path.join(base_path, file_path), "r") as f:
            obj = f.read()
    elif obj_type == "pickle":
        with open(os.path.join(base_path, file_path), "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Invalid obj_type {obj_type}")

    return obj


def safe_divide(a, b):
    return a / b if b != 0 else 0