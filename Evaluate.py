import ast
import pickle
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from scipy.stats import binomtest
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve, auc

#from Visualizer import plot_confusion_matrix, plot_all_rocs


def load_best_model_and_data(model_path, data_path, label_path):
    """
    Evaluate best model
    """
    # Laden des Modells
    #model_path = os.path.join(save_dir, f"saved_model.h5")
    model = tf.keras.models.load_model(model_path)
    #print(f"Modell {model_path} erfolgreich geladen!")

    # Laden der AUC und ROC-Daten
    #auc_file = os.path.join(save_dir, "best_auc_info.pkl")
    auc_data = joblib.load("results/final_results/synthetic_training/best_auc_info.pkl")
    fpr = np.array(auc_data["fpr"])  # False Positive Rate
    tpr = np.array(auc_data["tpr"])  # True Positive Rate
    thresholds = np.array(auc_data["thresholds"])  # Schwellenwerte
    auc_value = auc_data["auc"]  # AUC Wert
    #fold_saved = auc_data["fold"]

    print(f"AUC des besten Modells: {auc_value:.4f}")
    print(f"ROC-Daten: FPR {fpr}, TPR {tpr}, Schwellenwerte {thresholds}")
    #print(f"Gespeicherter Fold: {fold_saved}")

    # Testdaten und Testlabels laden
    #test_data_file = os.path.join(save_dir, f"test_data_saved_model.npy")
    #test_labels_file = os.path.join(save_dir, f"test_labels_saved_model.npy")
    test_data = np.load(data_path)
    test_labels = np.load(label_path)

    print(
        f"Succesful data loaded (Shape Data: {test_data.shape}, Shape Label: {test_labels.shape})")

    return model, auc_value, fpr, tpr, thresholds, test_data, test_labels

def unpack_all(path="results/fold_metrics.pkl"):
    # Initialisiere Listen für die Metriken
    train_accuracies, test_accuracies = [], []
    train_losses, test_losses = [], []
    roc_aucs = []
    total_conf_matrix = np.zeros((2, 2))  # Annahme: 2x2 Confusion Matrix
    history_list = []
    all_fpr, all_tpr = [], []

    # Lade die Pickle-Datei
    with open(path, "rb") as file:
        all_folds_metrics = pickle.load(file)

    # Gehe durch alle Folds und extrahiere die Metriken
    for fold, metrics_dict in all_folds_metrics.items():
        # Metriken extrahieren und in die Listen einfügen
        train_accuracies.append(metrics_dict["train_acc"])
        test_accuracies.append(metrics_dict["test_acc"])
        train_losses.append(metrics_dict["train_loss"])
        test_losses.append(metrics_dict["test_loss"])
        roc_aucs.append(metrics_dict["test_auc"])

        # Confusion Matrix (summiere über alle Folds)
        total_conf_matrix += metrics_dict["confusion matrix"]

        # FPR und TPR für ROC-Kurve
        all_fpr.append(metrics_dict["fpr"])
        all_tpr.append(metrics_dict["tpr"])

    # Hier kannst du weitere Verarbeitung oder Berechnungen durchführen, wenn nötig
    #print(f"TotalConfusion Matrix all Folds:\n{total_conf_matrix}")

    # Rückgabe der extrahierten Listen und Metriken
    return {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "roc_aucs": roc_aucs,
        "total_conf_matrix": total_conf_matrix,
        "all_fpr": all_fpr,
        "all_tpr": all_tpr
    }

def evaluate_all(path, include_best=True):
    metrics = unpack_all(path)

    train_accuracies = metrics["train_accuracies"]
    test_accuracies = metrics["test_accuracies"]
    train_losses = metrics["train_losses"]
    test_losses = metrics["test_losses"]
    roc_aucs = metrics["roc_aucs"]
    total_conf_matrix = metrics["total_conf_matrix"]
    all_fpr = metrics["all_fpr"]
    all_tpr = metrics["all_tpr"]

    model_path = "results/results/saved_model/saved_model.h5"
    data_path = "results/results/saved_model/test_data_saved_model.npy"
    label_path = "results/results/saved_model/test_labels_saved_model.npy"

    _, auc_value, fpr, tpr, thresholds, _,_ = load_best_model_and_data(model_path,data_path,label_path)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    roc_aucs.append(auc_value)

    #plot_all_rocs(all_fpr, all_tpr)
    conf_matrix_percent = total_conf_matrix.astype('float') / total_conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    #plot_confusion_matrix(conf_matrix_percent, class_names=["SZ", "BP"])

    best_auc = max(roc_aucs)
    worst_auc = min(roc_aucs)

    # Beste und schlechteste Accuracy
    best_acc = max(test_accuracies)
    worst_acc = min(test_accuracies)

    return conf_matrix_percent, all_fpr, all_tpr, roc_aucs, test_accuracies



# path = "results/fold_metrics_CNN2D.pkl"
#path = "results/final_results/synthetic_training/fold_metrics_CNN_combined.pkl"
#path = "results/final_results/original_training/fold_metrics_DNN.pkl"
# evaluate_all(path)

def p_val(model1_path, model2_path):
    _,_,_, auc_m1, _ = evaluate_all(model1_path)
    _, _, _, auc_m2, _ = evaluate_all(model2_path)

    cnn_wins = sum(c > d for c, d in zip(auc_m1, auc_m2))
    n = len(auc_m1)

    # Binomialtest (H0: beide gleich gut → p=0.5)
    result = binomtest(k=cnn_wins, n=n, p=0.5, alternative='two-sided')
    print("Binomialtest p-Wert:", result.pvalue)
    return result

# path1 = "results/final_results/original_training/fold_metrics_CNN_combined.pkl"
# path2 = "results/final_results/original_training/fold_metrics_CNN_pearson.pkl"
#
# p_val(path1,path2)

def confusion_of_model(model, test_data, test_labels):
    predictions = (model.predict(test_data) > 0.5).astype(int)
    conf_matrix = confusion_matrix(test_labels, predictions)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    print(conf_matrix_percent)

def get_active_rois(model, test_data, layer="conv2d", sample_index=1, threshold=0.7, mode="percentile", verbose=True):
    """
    Gibt die aktivsten ROIs basierend auf den absoluten Aktivierungen zurück.

    Args:
        model: Das trainierte Keras-Modell.
        test_data: Testdaten (Numpy-Array), shape z.B. (N, H, W, C)
        layer: Name des Convolutional Layers, aus dem die Aktivierungen extrahiert werden.
        sample_index: Index des Test-Samples.
        threshold: Schwellenwert für Aktivität (abhängig von `mode`)
        mode: 'percentile' (z. B. top 10%) oder 'absolute' (z. B. > 50 Aktivierung)
        verbose: Wenn True, druckt die aktiven ROI-Indizes.

    Returns:
        Eine Liste der aktiven ROI-Indizes.
    """
    input_sample = np.expand_dims(test_data[sample_index], axis=0)
    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    activations = activation_model.predict(input_sample)

    # Format sicherstellen: (ROIs, Filter)
    activation_maps = np.squeeze(activations, axis=0)
    if activation_maps.ndim == 3:
        if activation_maps.shape[0] == 1:
            activation_maps = np.squeeze(activation_maps, axis=0)
        elif activation_maps.shape[1] == 1:
            activation_maps = np.squeeze(activation_maps, axis=1)

    abs_activations = np.abs(activation_maps)
    col_sums = np.sum(abs_activations, axis=1)  # Summe über Filter → Relevanz pro ROI

    # Threshold anwenden
    if mode == "percentile":
        thresh_value = np.percentile(col_sums, threshold * 100)
        active_rois = np.where(col_sums >= thresh_value)[0]
    elif mode == "absolute":
        active_rois = np.where(col_sums >= threshold)[0]
    else:
        raise ValueError("Mode must be 'percentile' or 'absolute'")

    if verbose:
        print(f"Aktive ROIs (Threshold {threshold} - Mode: {mode}):", active_rois.tolist())

    return active_rois


