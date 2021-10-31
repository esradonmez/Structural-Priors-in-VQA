import os
import wandb

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import random
from tqdm import tqdm 

from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

from concept_classifier_dataloader import prepare_dataloader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
                              datefmt='%m/%d %H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--num_workers', default=56, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default=20, type=int, help = 'Number of training epochs')
parser.add_argument('--iou_alpha', default= 0.5, type=float)
parser.add_argument('--feature_emb', default=2048, type=int)
parser.add_argument('--hidden_size', default=2048, type=int)
parser.add_argument('--num_layers', default=6, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--lmda', default=1, type=float, help='weigh the concept loss lower?')
parser.add_argument('--output_dir', default='concept_classifier/checkpoints', type=str, help='')

class ConceptClassifier(nn.Module):

    def __init__(self, concepts, n_emb, n_hidden=512, num_layers=3, dropout=0.2):
        super().__init__()
        # here assume that concepts = [concept_val len ]
        self.main_head =  ClassificationHead(n_emb, n_hidden, num_layers, dropout, n_emb)
        self.object_head = nn.Linear(n_hidden, 2)
        self.heads = nn.ModuleList([ nn.Linear(n_hidden, cc_len) for cc_len in concepts])
        self.loss_fns = nn.ModuleList( nn.CrossEntropyLoss() for _ in range(len(concepts)+1))

    def forward(self, x, object_label, concept_labels):

        output_logits = {}
        output_loss = {}

        x = self.main_head(x) # transform
        object_logit = self.object_head(x)
        output_logits['object'] = object_logit
        output_loss['object'] = self.loss_fns[-1](object_logit.view(-1,(object_logit.size(-1))), object_label.view(-1))
        for cc_idx, head in enumerate(self.heads):
            logit = head(x)
            labels = concept_labels[:,cc_idx]
            output_logits[cc_idx] = logit
            output_loss[cc_idx] = self.loss_fns[cc_idx](logit.view(-1,(logit.size(-1))), labels.view(-1)) 
        return output_logits, output_loss

class ClassificationHead(nn.Module):
    def __init__(self, n_emb, n_hidden, num_layers=3, dropout=0.1, n_out=1):
        super().__init__()

        self.seq = nn.ModuleList([ nn.Sequential(
                nn.LayerNorm(n_hidden),
                nn.Dropout(dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU()) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.seq[:-1]:
            x = layer(x) + x
        x = self.seq[-1](x)
        return x 

def main(args):

    # start a W&B run
    wandb.init(project='cc')

    # save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = args.learning_rate
    config.num_layers = args.num_layers
    config.dropout = args.dropout

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    set_seed(42)

    print('Loading the dataset ...')
    train_dataloader, val_dataloader = prepare_dataloader(
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            num_workers = args.num_workers,
                                            drop_last = True, 
                                            iou_alpha = args.iou_alpha)

    print('Preparing concepts ...')
    concepts = [0]*train_dataloader.dataset.ccNum
    for k,v  in train_dataloader.dataset.concept2val.items():
        concepts[k] = v

    print('Initializing model ...')
    model = ConceptClassifier(concepts, args.feature_emb, args.hidden_size, args.num_layers, args.dropout)

    if torch.cuda.is_available() and args.use_cuda:
        print('use_cuda=True')
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
    
    print('Initializing optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    #logger.info("Batch size per GPU = %d", args.batch_size)

    #logger.info("Training...")
    model.zero_grad()
    for epoch in tqdm(range(args.num_epochs), desc='epochs'):
        #logger.info("Epoch: %5d", epoch)

        total_obj = 0.
        correct_obj = 0.
        total_cc = 0.
        correct_cc = 0.

        total_loss = 0.
        total_concept_loss = 0.

        for step, (feature, obj_label, cc_label) in tqdm(enumerate(train_dataloader)):
            object_loss = 0.
            concept_loss = 0.

            if args.use_cuda:
                feature = feature.to(device)
                obj_label = obj_label.to(device)
                cc_label = cc_label.to(device)

            output_logit, output_loss = model(feature, obj_label, cc_label) 

            loss = torch.tensor(0.)
            for k, v in output_loss.items():
                v = v.cpu()
                if k == 'object':
                    loss += v
                    object_loss = v.mean().item() 
                else:
                    loss += args.lmda*v
                    concept_loss += v.mean().item()         

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += object_loss
            total_concept_loss += concept_loss

        total_loss_val = 0.
        total_concept_loss_val = 0.

        #logger.info("Evaluating...")
        model.eval()
        for step, (feature, obj_label, cc_label) in tqdm(enumerate(val_dataloader)):
            object_loss = 0.
            concept_loss = 0.

            if args.use_cuda:
                feature = feature.to(device)
                obj_label = obj_label.to(device)
                cc_label = cc_label.to(device)
                    
            with torch.no_grad():
                output_logit, output_loss = model(feature, obj_label, cc_label)
                
            loss = torch.tensor(0.)
            for k, v in output_loss.items():
                v = v.cpu()
                if k == 'object':
                    loss += v
                    object_loss = v.mean().item() 
                else:
                    loss += args.lmda*v
                    concept_loss += v.mean().item()
            i = 0        
            for k, v in output_logit.items():
                v = v.cpu()
                if k == 'object':
                    pred = torch.argmax(torch.softmax(v, 1), 1).detach().cpu()
                    label = obj_label.cpu()
                    total_obj += 1
                    correct_obj += (pred == label).sum().item() / pred.size(0)
                else:
                    _, pred = torch.max(v, 1)
                    label = cc_label[:,k].cpu()
                    mask = label != 0
                    indices = torch.nonzero(mask)
                    label = label[indices]
                    pred = pred[indices]
                    total_cc += 1
                    correct_cc += (label == pred).sum().item() / pred.size(1)
                
            total_loss_val += object_loss
            total_concept_loss_val += concept_loss
        
        #logger.info("Avg training loss: %.5f", total_loss/len(train_dataloader.dataset))
        #logger.info("Avg concept training loss: %.5f", total_concept_loss/len(train_dataloader.dataset))


        # log metrics
        wandb.log({"Avg training loss obj": total_loss/len(train_dataloader)})
        wandb.log({"Avg training loss cc": total_concept_loss/len(train_dataloader)})
        wandb.log({"Avg val loss obj": total_loss_val/len(val_dataloader)})
        wandb.log({"Avg val loss cc": total_concept_loss_val/len(val_dataloader)})

        wandb.log({"Acc object": (100 * correct_obj / total_obj)})
        wandb.log({"Acc concept": (100 * correct_cc / total_cc)})

        #if epoch % 10 == 0 and epoch != 0 :
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch)) )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)