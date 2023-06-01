from copy import deepcopy
import torch
import numpy as np


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=20):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloaders[phase], 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                # print()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 1.0 * running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())
            else:
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

        print()

    print('Best val Acc: {:4f}'.format(best_acc))
    history = np.vstack((train_loss_history, train_acc_history, val_loss_history, val_acc_history))

    return best_model_wts, history


def infer(model, dataloader, device, embedding_dim, testing=True):
    model.eval()
    embeddings = np.empty((0, embedding_dim))
    targets = []
    predictions = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)

        outputs = model.get_embedding(inputs)
        norm = torch.norm(outputs, dim=1, keepdim=True)
        vectors = outputs / norm

        embeddings = np.vstack((embeddings, vectors.detach().cpu().numpy()))
        targets += list(labels.detach().cpu().numpy())

        if not testing:
            outputs = model.get_similarity(outputs)
            _, preds = torch.max(outputs, 1)
            predictions += list(preds.detach().cpu().numpy())

    if testing:
        return embeddings, targets
    else:
        return embeddings, targets, predictions


def get_features(model, samples, device):
    model.eval()
    features = model.get_features(samples.to(device))

    return features.detach().cpu().numpy()
