import os
import os.path as osp
import json
import shutil
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
from PIL import Image
from lib.learning.network import Network
from lib.learning.trainer import train_model, infer, get_features
from lib.learning.dataset import get_class_weights, ImageSet
from lib.analysis.visualization import plot_history, plot_embeddings, plot_feature_maps
from lib.analysis.clustering import clustering, aggregation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.close('all')

    # common
    t = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    batch_size = 100
    embedding_dim = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained = True
    # pretrained = False
    keep_record = True

    #
    # Train
    #

    root = osp.join('data', 'train')
    dataset = ImageFolder(root, t)
    num_classes = len(dataset.classes)

    # train-val split & data loaders
    train_set, val_set = random_split(dataset, (0.8, 0.2))
    dataloaders = dict()
    dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # model
    model = Network(num_classes, embedding_dim)
    model.to(device)

    # training
    learning_result_root = osp.join('results', 'learning')
    weight_file = osp.join(learning_result_root, 'model_state_dict.pt')
    history_file = osp.join(learning_result_root, 'training_statistics.txt')
    if pretrained:
        best_model_wts = torch.load(weight_file)
        history = np.loadtxt(history_file)
    else:
        # criterion
        criterion = nn.CrossEntropyLoss()
        # # class weights for classification balance
        # class_weights = get_class_weights(dataset)
        # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_model_wts, history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=20)

        if keep_record:
            if not osp.exists(learning_result_root):
                os.makedirs(learning_result_root)
            torch.save(best_model_wts, weight_file)
            np.savetxt(history_file, history, '%.4f')

    model.load_state_dict(best_model_wts)

    #
    # Analysis
    #

    # training history
    plot_history(history)

    # embeddings
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # entire training set
    embeddings, labels, predictions = infer(model, dataloader_train, device, embedding_dim, testing=False)

    # confusion matrix
    mat = confusion_matrix(y_true=labels, y_pred=predictions)
    df_mat = pd.DataFrame(mat)

    # embedding visualization (T-SNE)
    pd.MultiIndex.from_tuples
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    cls = [idx_to_class[lb] for lb in labels]
    plot_embeddings(embeddings, cls)

    # feature maps
    n_sample = 3
    np.random.seed(42*n_sample)
    sample_idx = np.random.choice(len(dataset), n_sample, replace=False)
    sample_dataset = Subset(dataset, sample_idx)
    dataloader_sample = DataLoader(sample_dataset, batch_size=n_sample, shuffle=False)
    samples, _ = next(iter(dataloader_sample))
    features = get_features(model, samples, device)
    plot_feature_maps(samples.numpy(), features)

    #
    # Test
    #

    root_test = osp.join('data', 'test')
    im_name = osp.join(root_test, 'shelf.jpeg')
    det_name = osp.join(root_test, 'shelf.json')

    with Image.open(im_name) as im:
        with open(det_name) as det_file:
            det = json.load(det_file)
            ids = list(det.keys())
            bbs = [(v['x1'], v['y1'], v['x2'], v['y2']) for v in det.values()]
            det_ims = [im.crop(bb) for bb in bbs]
            id_to_idx = {v: k for k, v in enumerate(ids)}

            # dataset and data loader
            dataset_test = ImageSet(det_ims, [id_to_idx[i] for i in ids], t)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            embeddings, embedding_ids = infer(model, dataloader_test, device, embedding_dim, testing=True)

            # clustering
            clusters = clustering(embeddings)

            # saving the clusters
            if keep_record:
                cluster_result_root = osp.join('results', 'clustering')
                if osp.exists(cluster_result_root):
                    shutil.rmtree(cluster_result_root)
                for i in range(len(clusters)):
                    label = clusters[i]
                    c_folder = osp.join(cluster_result_root, str(label))
                    if not osp.exists(c_folder):
                        os.makedirs(c_folder)
                    det_ims[i].save(osp.join(c_folder, ids[i] + '.jpg'))

            # aggregation visualization
            aggregation(im, bbs, clusters)

    if keep_record:
        plt.waitforbuttonpress()
