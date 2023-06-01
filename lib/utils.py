import numpy as np


def cosine_distance(embeddings):
    sim = np.matmul(embeddings, np.transpose(embeddings))
    dist = np.maximum(1.0 - sim, 0.0)
    return dist
