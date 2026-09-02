"""
Step 2 of the pipeline: train the PyTorch classifier on the landmark CSVs
collected in data/.

Run:
    python src/train_model.py

Outputs:
    models/asl_model.pth   -> trained weights (used by BOTH the desktop
                               demo and the FastAPI backend)
    models/labels.json     -> index-to-label mapping

Prints train/test accuracy and a per-class report so you have real numbers
to put on your resume once you've collected your own data.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import config
from model import ASLClassifier


def load_dataset():
    csv_files = glob.glob(os.path.join(config.DATA_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {config.DATA_DIR}. "
            "Run data_collection.py for each gesture first."
        )

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes "
          f"from {len(csv_files)} files.")
    print(df["label"].value_counts())
    return df


def main():
    df = load_dataset()

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = ASLClassifier(input_dim=X.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 100
    BATCH_SIZE = 32
    n_train = X_train_t.shape[0]

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = accuracy_score(y_test_t, preds)
            print(f"Epoch {epoch:3d} | loss: {epoch_loss / n_train:.4f} | test acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()

    print("\n=== Final classification report ===")
    print(classification_report(y_test, preds, target_names=encoder.classes_))
    final_acc = accuracy_score(y_test, preds)
    print(f"Final test accuracy: {final_acc:.4f}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X.shape[1], "num_classes": num_classes},
        config.MODEL_PATH,
    )
    with open(config.LABELS_PATH, "w") as f:
        json.dump(list(encoder.classes_), f)

    print(f"\nSaved model to {config.MODEL_PATH}")
    print(f"Saved label map to {config.LABELS_PATH}")
    print(f"\n>>> Resume-ready metric: {final_acc*100:.1f}% test accuracy "
          f"across {num_classes} gesture classes <<<")
    print("\nThis model now works with BOTH:")
    print("  - Desktop demo:  python src/recognize_desktop.py")
    print("  - Web app:       cd webapp/backend && uvicorn main:app --reload")


if __name__ == "__main__":
    main()
