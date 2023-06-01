import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, n_cls, embedding_dim=64):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                                      nn.Conv2d(4, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.embedding = nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features=576, out_features=embedding_dim, bias=True),
                                       nn.Sigmoid(),
                                       nn.LayerNorm(embedding_dim))

        anchors = torch.rand(1, n_cls, embedding_dim, requires_grad=True) - 0.5
        self.anchors = nn.Parameter(anchors, requires_grad=True)
        self.similarity = nn.CosineSimilarity(dim=2)

    def get_features(self, x):
        x = self.features(x)
        return x

    def get_embedding(self, x):
        x = self.get_features(x)
        x = self.pool(x)
        mbd = self.embedding(x)

        return mbd

    def get_similarity(self, mbd):
        sim = self.similarity(mbd[:, None, :], self.anchors)
        return sim

    def forward(self, x, lmd=3.0):
        mbd = self.get_embedding(x)
        sim = self.get_similarity(mbd)
        return sim * lmd
