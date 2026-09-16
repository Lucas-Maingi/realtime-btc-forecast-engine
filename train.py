import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from features import FEATURE_COLUMNS
from model import BTCPriceClassifier


def load_and_split_data(csv_path="btc_features.csv"):
    """
    Loads features and performs a strict 3-way chronological split:
    70% Train, 15% Validation, 15% Test.
    """
    df = pd.read_csv(csv_path)

    x = df[FEATURE_COLUMNS].values
    y = df["target"].values.astype(np.float32)

    total_rows = len(df)
    train_end = int(total_rows * 0.70)
    val_end = int(total_rows * 0.85)

    # Chronological slicing (no shuffle!)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    print(f"Data split:")
    print(f"  Train: {len(x_train)} rows")
    print(f"  Val:   {len(x_val)} rows")
    print(f"  Test:  {len(x_test)} rows")

    # Fit scaler ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Save scaler for live deployment
    joblib.dump(scaler, "scaler.joblib")
    print("Saved feature scaler to scaler.joblib")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_dataloader(x, y, batch_size=64, shuffle=False):
    """Converts numpy arrays into PyTorch DataLoader."""
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # shape: (N, 1)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(epochs=25, batch_size=64, lr=0.001):
    # 1. Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()

    train_loader = create_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(x_test, y_test, batch_size=batch_size, shuffle=False)

    # 2. Instantiate model, loss function, and optimizer
    model = BTCPriceClassifier(input_dim=len(FEATURE_COLUMNS))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_x)

        avg_train_loss = total_train_loss / len(x_train)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                total_val_loss += loss.item() * len(batch_x)

                # Convert probability > 0.5 to class 1 (UP), else 0 (DOWN)
                predicted_class = (preds >= 0.5).float()
                correct_predictions += (predicted_class == batch_y).sum().item()

        avg_val_loss = total_val_loss / len(x_val)
        val_accuracy = (correct_predictions / len(x_val)) * 100

        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "model.pth")
            saved_marker = " --> Model saved (Best val loss)"
        else:
            saved_marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%{saved_marker}")

    # 3. Final Test Evaluation (The Final Exam)
    print("\n--- Evaluating on Unseen Test Set ---")
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            predicted_classes = (preds >= 0.5).float()
            test_correct += (predicted_classes == batch_y).sum().item()

    test_accuracy = (test_correct / len(x_test)) * 100
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()
