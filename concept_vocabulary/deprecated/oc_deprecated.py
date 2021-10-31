import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch import models
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import precision_score

class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50_32x4d(pretrained=True)

        # change resnet.fc to different number of classes
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )

        self.base_model = resnet
        # for multilabel classification, use sigmoid at output layer
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples')}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train():
    batch_size = 32
    max_epoch_num = 35
    iteration = 0
    epoch = 0
    learning_rate = 1e-3
    
    set_seed(42)

    # TODO: correct n_classes
    n_classes = 77

    # TODO: define save_freq
    save_freq = 2
    
    model = Resnext50(n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # for multilabel classification, use binary cross entropy
    criterion = nn.BCELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    model.to(device)

    while True:
        batch_losses = []
        for features, targets in train_dataloader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()

            model_output = model(features)
            loss = criterion(model_output, targets.type(torch.float))

            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss_value)

            for iteration % test_freq == 0:
                model.eval()
                with torch.no_grad():
                    model_output = []
                    targets = []
                    for features, batch_targets in test_dataloader:
                        features = features.to(device)
                        model_batch_output = model(features)
                        model_output.extend(model_batch_output.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())
                
                results = calculate_metrics(np.array(model_output),
                                            np.array(targets))
                print("epoch:{:2d} iter:{:3d} test: " 
                    "micro f1: {:.3f} "
                    "macro f1: {:.3f} "
                    "samples f1: {:.3f}".format(epoch, iteration,
                                                result['micro/f1'],
                                                result['macro/f1'],
                                                result['samples/f1']))
                
                model.train()
            iteration += 1

        loss_value = np.mean(batch_losses)
        print("epoch:{:2d} iter:{:3d} train: loss:{:3f}".format(epoch,
                                                                iteration,
                                                                loss_value))

        if epoch % save_freq == 0:
            checkpoint_save(model, save_path, epoch)
        
        epoch += 1

        if max_epoch_num < epoch:
            break

if __name__ == "__main__":
    train()