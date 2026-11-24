"""52-channel residual encoder for the WiSig unseen-receiver experiment."""
from torch import nn


class ResidualUnit(nn.Module):
    def __init__(self, source, target, kernel):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv1d(source, target, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(target), nn.ReLU(),
            nn.Conv1d(target, target, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(target))
        self.shortcut = nn.Sequential(nn.Conv1d(source, target, 1, bias=False), nn.BatchNorm1d(target))
        self.activation = nn.ReLU()

    def forward(self, value):
        return self.activation(self.path(value) + self.shortcut(value))


class ReceiverClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [52, 32, 64, 128, 256]
        self.stages = nn.ModuleList([ResidualUnit(widths[i], widths[i + 1], kernel)
                                    for i, kernel in enumerate([7, 5, 5, 3])])
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2560, 512), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(512, 6))

    def forward(self, rows):
        value = rows.transpose(1, 2)
        for stage in self.stages:
            value = self.pool(stage(value))
        return self.head(value)
