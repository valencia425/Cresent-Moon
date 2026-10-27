# Binary Classification Using the make_moons Dataset

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 65)
print("ANN BINARY CLASSIFICATION USING MAKE_MOONS")
print("=" * 65)


# 2. GENERATE DATASET (10,000 SAMPLES)
X, y = make_moons(
    n_samples=10000,
    noise=0.20,
    random_state=42
)

# Create a DataFrame
df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)
df["Target"] = y

print("\nDATASET INFORMATION")
print("-" * 45)
print("Total samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

print("\nFirst 5 records:")
print(df.head())

print("\nClass distribution:")
print(df["Target"].value_counts().sort_index())


# 3. VISUALIZE DATASET
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0",
    alpha=0.5
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1",
    alpha=0.5
)

plt.title("make_moons Dataset - 10,000 Samples")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 4. SPLIT DATASET INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATA SPLITTING")
print("-" * 45)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. STANDARDIZE INPUT FEATURES
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature standardization completed successfully.")


# 6. BUILD THE ARTIFICIAL NEURAL NETWORK
model = Sequential([
    Input(shape=(2,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the ANN
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nANN MODEL ARCHITECTURE")
model.summary()


# 7. TRAIN THE ANN
print("\nTRAINING ANN MODEL...")

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)

print("\nANN training completed successfully!")


# 8. EVALUATE THE MODEL
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# Generate predictions
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-" * 45)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")


# 9. CLASSIFICATION REPORT
print("\nCLASSIFICATION REPORT")
print("-" * 45)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"],
        zero_division=0
    )
)


# 10. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"]
)

disp.plot()
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm)


# 11. PLOT TRAINING AND VALIDATION ACCURACY AND LOSS
plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

# Loss graph
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 12. VISUALIZE THE DECISION BOUNDARY
# Use the original feature coordinates for plotting
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Prepare the grid
grid = np.c_[xx.ravel(), yy.ravel()]

# Standardize grid using the fitted scaler
grid_scaled = scaler.transform(grid)

# Predict class probabilities
Z = model.predict(
    grid_scaled,
    verbose=0
).ravel()

# Convert probabilities into class labels
Z = (Z >= 0.5).astype(int).reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(9, 7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="coolwarm"
)

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0",
    alpha=0.5
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1",
    alpha=0.5
)

plt.title("ANN Decision Boundary - 10,000 Samples")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()


# 13. PREDICT A NEW SAMPLE
sample = np.array([[0.5, 0.2]])

# Standardize the sample
sample_scaled = scaler.transform(sample)

# Predict probability
probability = model.predict(
    sample_scaled,
    verbose=0
)[0][0]

# Convert probability into a class
predicted_class = int(probability >= 0.5)

print("\nSAMPLE PREDICTION")
print("-" * 45)
print("Input features:", sample[0])
print(f"Probability of Class 1: {probability:.4f}")
print(f"Predicted Class: {predicted_class}")


# 14. SAVE THE DATASET AS A CSV FILE
dataset = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)

dataset["Target"] = y

dataset.to_csv(
    "make_moons_dataset.csv",
    index=False
)

print("\nDataset saved as make_moons_dataset.csv")


# 15. DOWNLOAD DATASET IN GOOGLE COLAB
try:
    from google.colab import files
    files.download("make_moons_dataset.csv")
except ImportError:
    print("Not running in Google Colab.")
    print("The CSV file is saved in the current directory.")


# 16. FINAL RESULT
print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Dataset: make_moons")
print("Total Samples: 10,000")
print("Input Features: 2")
print("Output Classes: 2")
print("ANN Hidden Layers: 32, 16, 8 neurons")
print("Training Epochs: 150")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("Dataset File: make_moons_dataset.csv")
print("=" * 65)
