import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLREncoder(nn.Module):
    """
    SimCLR encoder for 19-channel EEG windows.

    Expected input shape:
        (batch_size, 19, time_samples)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=19,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projector(features)
        projection = F.normalize(projection, dim=1)

        return projection


class SimCLRClassifier(nn.Module):
    """
    Binary normal-versus-abnormal EEG classifier built on a
    pretrained SimCLR encoder.
    """

    def __init__(self, pretrained_encoder):
        super().__init__()

        self.encoder = pretrained_encoder.encoder

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)

        return logits
