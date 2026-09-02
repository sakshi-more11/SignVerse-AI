"""
A small feed-forward neural network that classifies a 63-dim hand-landmark
feature vector into one of the gesture classes.

This is the modern replacement for classical SVM+HOG classifiers used in
older sign-language recognition projects. A landmark-based MLP is the
standard approach for static gesture recognition — it trains in seconds on
CPU, is tiny (~50KB), and is far more robust than pixel-based classifiers
because it operates on hand geometry rather than raw image texture or skin
color.

This exact architecture is used by both the desktop pipeline (src/) and the
FastAPI backend (webapp/backend/), so a model trained once with
src/train_model.py works in both.
"""
import torch
import torch.nn as nn


class ASLClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
