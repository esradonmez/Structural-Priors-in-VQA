import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_schedular
import torch.nn.functional as F

from torch import nn
from torch import models
from torch.utils.data import DataLoader, Dataset

from tqdm import trange

from sklearn.metrics import precision_score, f1_score

class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50_32x4d(pretrained=True)

        num_features = resnet.fc.in_features

        # freeze the conv base
        for param in resnet.parameters():
            param.requires_grad_(False)

        self.top_head = self.create_head(num_features, len(n_classes))
        # replace the fully-connected layer
        resnet.fc = top_head

        self.model = resnet

    def forward(self, x):
        return self.model(x)
    
    def create_head(num_features, n_classes, dropout_prob=0.5, actication_func=nn.ReLU):
        feature_lst = [num_features, num_features//2, num_features//4]

        layers = []
        for in_f, out_f in zip(feature_lst[:-1], feature_lst[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(actication_func)
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob !=0:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(feature_lst[-1], n_classes))

        return nn.Sequential(*layers)
            

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

def train(dataloader, num_epochs=5):
    set_seed(42)
    learning_rate = 1e-3
    # TODO: correct n_classes
    n_classes = 77
    
    model = Resnext50(n_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    schedular = lr_schedular.CosineAnnealignLR(optimizer, T_max=5, eta_min=0.005)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'val']:
            if phase=='train': # put the model in training mode
            model.train()
            schedular.step()
        else: # put the model in validation mode
            model.eval()
        
        # keep track of training and validation loss
        running_loss = 0.0
        running_corrects = 0.0

        for data, target  in dataloader[phase]:
            # load the data and target to respective device
            data, target = data.to(device), target.to(device)

            with torch.set_grad_enabled(phase='train'):
                # feed the input
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                preds = torch.sigmoid(output).data > 0.5
                preds = preds.to(torch.float32)

                if phase=='train':
                    loss = backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            # statistics
            running_loss += loss.item() * data.size(0)
            running_corrects += f1_score(target.to("cpu").to(torch.int).numpy(),
                                        preds.to("cpu").to(torch.int).numpy(),
                                        average="samples" * data.size(0))
            
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects / len(dataloader[phase].dataset)

            result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print(result)

if __name__ == "__main__":
    train()