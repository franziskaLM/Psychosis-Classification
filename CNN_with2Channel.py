import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Constant
from tensorflow.keras.callbacks import EarlyStopping
# from Preprocessor_2Channel import (
#     load_data,
#     mixup_data,
#     combine_and_balance
# )
from Preprocesor import load_data, mixup_data, combine_and_balance
from Safe_results import safe_model, log_fold_results_to_csv, log_fold_results_to_pickle
from Visualizer import plot_roc, plot_confusion_matrix, plot_all_rocs

# --- Parameters ---
correlation_type = "combined" #pearson & partial

number_folds = 50

use_synthetic_train_data = True
num_synthetic = 0


# --- Model Def---
def build_cnn_model(input_shape=(105, 105, 2), num_hidden=96, num_labels=1, depth=64):
    model = Sequential()
    model.add(Conv2D(filters=depth, kernel_size=(105, 1), activation='relu',
                     input_shape=input_shape, padding='valid',
                     kernel_initializer=GlorotUniform(), bias_initializer=Constant(0.001)))

    model.add(Conv2D(filters=depth * 2, kernel_size=(1, 105), activation='relu',
                     padding='valid', kernel_initializer=GlorotUniform(), bias_initializer=Constant(0.001)))

    model.add(Flatten())

    model.add(Dense(units=num_hidden, activation='relu',
                    kernel_initializer=GlorotUniform(), bias_initializer=Constant(0.01)))
    model.add(Dense(units=num_labels, activation='sigmoid',
                    kernel_initializer=GlorotUniform(), bias_initializer=Constant(0.01)))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    return model


# --- Load and Augment Data ---

train_data, train_labels, test_data, test_labels, val_data, val_labels = load_data(
    correlation="combined", scale=True, Split=True, test_size=0.33 / 2, val_size=0.33 / 2
)

if use_synthetic_train_data:


    # Apply Mixup
    augmented_data, augmented_labels = mixup_data(train_data, train_labels, num_new_samples=num_synthetic)
    train_data_extended, train_labels_extended = combine_and_balance(train_data, train_labels, augmented_data,
                                                                     augmented_labels)
    train_data_extended, train_labels = shuffle(train_data_extended, train_labels_extended)

    # Ensure correct shape
    train_data = np.squeeze(np.expand_dims(train_data_extended, axis=-1), axis=-1)
    test_data = np.squeeze(np.expand_dims(test_data, axis=-1), axis=-1)
    val_data = np.squeeze(np.expand_dims(val_data, axis=-1), axis=-1)

else:
    train_data = np.squeeze(np.expand_dims(train_data, axis=-1), axis=-1)
    test_data = np.squeeze(np.expand_dims(test_data, axis=-1), axis=-1)
    val_data = np.squeeze(np.expand_dims(val_data, axis=-1), axis=-1)


# --- Training with Cross-Validation ---
from sklearn.metrics import roc_curve, auc

# --- Training with Cross-Validation ---
k_folds = number_folds
train_accuracies, test_accuracies = [], []
train_losses, test_losses = [], []
roc_aucs = []
total_conf_matrix = np.zeros((2, 2))
history_list = []
all_fpr, all_tpr = [], []

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

for fold in range(k_folds):

    print(f"\n--- Fold {fold + 1} ---")

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels),
                                         y=train_labels)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Build and train model
    model = build_cnn_model()
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        class_weight=class_weight_dict,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )
    history_list.append(history.history)

    # Evaluate performance
    train_loss, train_acc, train_auc = model.evaluate(train_data, train_labels, verbose=0)
    test_loss, test_acc, test_auc = model.evaluate(test_data, test_labels, verbose=0)

    print(f"Train AUC: {train_auc * 100:.2f}%")
    print(f"Test AUC: {test_auc * 100:.2f}%")

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    roc_aucs.append(test_auc)

    # Confusion Matrix
    predictions = (model.predict(test_data) > 0.5).astype(int)
    conf_matrix = confusion_matrix(test_labels, predictions)
    total_conf_matrix += conf_matrix
    print(conf_matrix)

    # ROC Curve (on the test set)
    y_pred_prob = model.predict(test_data)  # Vorhersagen der Wahrscheinlichkeit
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # Berechnung der AUC
    print(f"Test ROC AUC: {roc_auc:.4f}")

    #plot_roc(fpr, tpr,roc_auc,fold)
    all_fpr.append(fpr)
    all_tpr.append(tpr)

    metrics = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "confusion matrix": conf_matrix,
        "fpr":fpr,
        "tpr":tpr,
        "thresholds":thresholds
    }

    log_fold_results_to_pickle(fold, metrics, path="results/fold_metrics_CNN2D_5.pkl")
    safe_model(model, test_auc, fpr, tpr, thresholds, test_data, test_labels)

# --- Average Metrics ---
print("\n=== Average Metrics Across All Folds ===")
print(f"Avg Train Accuracy: {np.mean(train_accuracies) * 100:.2f}%")
print(f"Avg Train Loss: {np.mean(train_losses):.4f}")
print(f"Avg Test Loss: {np.mean(test_losses):.4f}")

print(f"Avg Test Accuracy: {np.mean(test_accuracies) * 100:.2f}% ± {np.std(test_accuracies):.4f}")
print(f"Avg Test AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")


# --- Normalized Confusion Matrix ---


conf_matrix_percent = total_conf_matrix.astype('float') / total_conf_matrix.sum(axis=1)[:, np.newaxis] * 100
print("Normalized Confusion Matrix (row-wise %):")
#print(conf_matrix_percent)
plot_confusion_matrix(conf_matrix_percent, class_names=["SZ", "BP"], title="CNN 2D")
plot_all_rocs(all_fpr, all_tpr)