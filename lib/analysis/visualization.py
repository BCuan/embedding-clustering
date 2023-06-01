import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from lib.utils import cosine_distance


def plot_history(history):
    plt.ion()

    num_epoch = np.shape(history)[1]

    df = pd.DataFrame()
    epochs = np.arange(num_epoch, dtype=int)
    df["Training Loss"] = history[0, :]
    df["Validation Loss"] = history[2, :]
    df["Training Accuracy"] = history[1, :]
    df["Validation Accuracy"] = history[3, :]

    plt.figure()
    g = sns.lineplot(data=df[['Training Loss', 'Validation Loss']], markers=True)
    g.set(title='Loss', xlabel='Epoch', ylabel='Loss', xticks=epochs)
    plt.show()
    # plt.pause(0.001)

    plt.figure()
    g = sns.lineplot(data=df[['Training Accuracy', 'Validation Accuracy']], markers=True)
    g.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy', xticks=epochs)
    plt.show()
    # plt.pause(0.001)


def plot_embeddings(embeddings, cls):
    plt.ion()

    # cosine similarity
    dist = cosine_distance(embeddings)

    tsne = TSNE(n_components=2, init='random', metric='precomputed')
    embeddings_tsne = tsne.fit_transform(dist)
    print("KL divergence: " + str(tsne.kl_divergence_))

    df = pd.DataFrame()
    df["y"] = cls
    df["comp-1"] = embeddings_tsne[:, 0]
    df["comp-2"] = embeddings_tsne[:, 1]

    num_cls = len(np.unique(cls))
    plt.figure()
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("colorblind", num_cls),
                    data=df).set(title="Embedding Visualization")

    plt.show()
    # plt.pause(0.001)


def plot_feature_maps(ims, features):
    plt.ion()

    assert len(ims) == len(features)
    g, axes = plt.subplots(2, len(ims))
    g.suptitle('Feature Maps')
    for i, (im, feat) in enumerate(zip(ims, features)):
        feat = np.mean(feat, axis=0, keepdims=False)

        axes[0, i].imshow(np.transpose(im, (1, 2, 0)))
        axes[0, i].axis('off')
        axes[1, i].imshow(feat)
        axes[1, i].axis('off')
    plt.show()
    # plt.pause(0.001)
