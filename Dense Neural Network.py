import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight, shuffle

from Safe_results import log_fold_results_to_csv, safe_model, log_fold_results_to_pickle
from Visualizer import visualize_fnc, plot_confusion_matrix, plot_all_rocs
from Analyser import ROC, CONFUSION
import matplotlib.pyplot as plt
from Preprocesor import load_data, mixup_data, combine_and_balance
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
import os


# --- Parameters ---

correlation_type = "partial" # Alternative:
#correlation_type = "pearson"
#correlation_type = "combined"

k_folds = 100
use_synthetic_train_data = True
num_synthetic = 100


def DNN(input_dim=5460):
    model = Sequential()

    # Input layer: 5460 input features
    model.add(Dense(128, input_dim=input_dim, activation='relu'))  # First hidden layer with 128 neurons
    model.add(Dense(64, activation='relu'))  # Second hidden layer with 64 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification: SZ or BP)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    return model

def DNN2(input_dim=5460):
    model = Sequential()
    # Input layer: 5460 input features
    model.add(Dense(256, input_dim=input_dim, activation='relu'))  # First hidden layer with 128 neurons
    model.add(Dropout(0.1))  # Dropout regularization
    model.add(Dense(128, activation='relu'))  # Second hidden layer with 64 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification: SZ or BP)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    return model

train_data, train_labels, test_data, test_labels, val_data, val_labels = load_data(correlation = correlation_type,
    scale=True, Split=True, test_size=0.33 / 2, val_size=0.33 / 2)

# --- Load and Augment Data ---
if use_synthetic_train_data:
    print(train_data.shape)
    # Apply Mixup
    augmented_data, augmented_labels = mixup_data(train_data, train_labels, num_new_samples=num_synthetic)
    train_data_extended, train_labels_extended = combine_and_balance(train_data, train_labels, augmented_data,
                                                                     augmented_labels)
    train_data_extended, train_labels = shuffle(train_data_extended, train_labels_extended)

    # Ensure correct shape
    train_data = np.squeeze(np.expand_dims(train_data_extended, axis=-1), axis=-1)
    test_data = np.squeeze(np.expand_dims(test_data, axis=-1), axis=-1)
    val_data = np.squeeze(np.expand_dims(val_data, axis=-1), axis=-1)


# --- Load only original Data ---
else:
    train_data = np.squeeze(np.expand_dims(train_data, axis=-1), axis=-1)
    test_data = np.squeeze(np.expand_dims(test_data, axis=-1), axis=-1)
    val_data = np.squeeze(np.expand_dims(val_data, axis=-1), axis=-1)
#Flatten
train_data = train_data.reshape((train_data.shape[0], -1))
test_data = test_data.reshape((test_data.shape[0], -1))
val_data = val_data.reshape((val_data.shape[0], -1))


# Visualize random samples from the train data
#visualize_fnc(train_data[11])




# --- Training with Cross-Validation ---
train_accuracies, test_accuracies = [], []
train_losses, test_losses = [], []
roc_aucs = []
total_conf_matrix = np.zeros((2, 2))
history_list = []

# Initialize variables to store all FPR and TPR for plotting ROC curves later
all_fpr, all_tpr, all_roc_auc = [], [], []

early_stopping = EarlyStopping(
    monitor='val_loss',
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

    print("Input Shape Model:", train_data.shape[1])
    model = DNN2(input_dim=train_data.shape[1])
    print("Created new model.")
    # Train the model
    history = model.fit(
            train_data,
            train_labels,
            validation_split = 0.2,
            class_weight=class_weight_dict,
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping]
        )
    history_list.append(history.history)

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
    y_pred_prob = model.predict(test_data)  # Probabilities for ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # Calculate AUC

    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Store the ROC data for later plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

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
    log_fold_results_to_pickle(fold, metrics, path="results/final_results/synthetic_training/fold_metrics_DNN_partial.pkl")



print("\n=== Average Metrics Across All Folds ===")
print(f"Avg Train Accuracy: {np.mean(train_accuracies) * 100:.2f}%")
print(f"Avg Train Loss: {np.mean(train_losses):.4f}")
print(f"Avg Test Loss: {np.mean(test_losses):.4f}")
print(f"Avg Test Accuracy: {np.mean(test_accuracies) * 100:.2f}% ± {np.std(test_accuracies):.4f}")
print(f"Avg Test AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")

# --- Normalized Confusion Matrix ---
conf_matrix_percent = total_conf_matrix.astype('float') / total_conf_matrix.sum(axis=1)[:, np.newaxis] * 100
print("Normalized Confusion Matrix (row-wise %):")
plot_confusion_matrix(conf_matrix_percent, class_names=["SZ", "BP"], title="DNN")

# --- Plot All ROC Curves ---
plot_all_rocs(all_fpr, all_tpr, )  # Function to plot all ROC curves with AUC values