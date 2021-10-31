'''
here we will define a trainig loop for fine-tuning a pre-trained object recognition 
for the defined concept vocabulary 

1. fine-tune Faster-RCNN using the spatial features 
2. fine-tune object classification model using the object features 

steps 
we have the spatial features from GQA - restnet 101 layer 
-- train a object classification model on the object features (take some pytorch - torchvision - take pretrained model like faster-rcnn and )

'''

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset

import os
import glob 
import h5py


class ObjectClassificationMultiClass(nn.Module):
    '''
    this model takes object features and uses object classification model 

    object_features
        object 1 - features : object name / object attributes / object relations 
        object 2 - features 
        object 3 - features 
        ...
        object n - features 

    x = object feature (true label : 'carrot' / 'orange', 'big' / 'in front' )
    y = one hot encoded vectors : say C concepts 
        C = 1 object name [0,...,1,...0]
        C = 2 color concept [0,..,1,...0] - red and orange 
        .
        .
        .
        C = C relation concept [0,...1,...0] 

    only caveat - what if our cluster was defined in a way that given (check mutually exclusive )
    '''
    def __init__(self, model):
        self.model # pretrained model 
        self.classifiers = nn.ModuleList[nn.Linear(10,2) for i in range(10)
        self.loss_fns = []

    def forward(self, x ,y):
        
        x = model(x)
        # y a tensor 

        if y is not None:
            loss = self.loss_fn(x,y)
            pass 
            # output logits and loss 
        #return logtis and loss\

        
def train():
    '''
    prepares h5py file format ... all the 

    json['object_id'] = matrix of [C, number of ]

    when you are training - track evaluation accuracy and loss
    '''

    pass

class FasterRCnnFineTune(nn.Module):
    '''
    this model takes spatial features and uses pretraiend faster-rcnn model to extract objects 
    classify them according to concept vocabulary
    '''
    def __init__(self, model):
        pass

# dataset = Dataset(data) # object that holds the data 
# '''
# DATASET
# self.data = [a,b,c,d....,n]
# def __len__():
#     return len(self.data)
# def __getitem__(self, idx):
#     return self.data[idx]


# DATALOADER 
# 1. define your own collate_fn
# batch_size =16

# random.choice(0,len(self.data)) # batch number of times 
# batched = [getitem(batch_idx_0), ...,getitem(batch_idx_n)]

# def collate_fn(batch):
#     prepare into tensors 

# '''
# dataloader = DataLoader(dataset=data ) # deals with batching and sampling ....

# class AlphaDataset(Dataset):
# 	def __init__(self,
# 				data_path,
# 				tokenizer, 
# 				max_samples=None):
# 		self.data = open_tsv_file(data_path, dic=True)
# 		self.tokenizer = tokenizer
# 		self.max_samples = max_samples

# 	def __len__(self):
# 		if self.max_samples is None:
# 			return len(self.data['obs1'])
# 		return self.max_samples

# 	def __getitem__(self, idx):

# 		items = {}
# 		items['hyp1'], items['hyp1_mask'], items['hyp1_reference'] = self.preprocess_hypothesis(self.data['hyp1'][idx])
# 		items['hyp2'], items['hyp2_mask'], items['hyp2_reference'] = self.preprocess_hypothesis(self.data['hyp2'][idx])

# 		observation = [self.data['obs1'][idx], self.data['obs2'][idx]]
# 		items['obs'], items['obs_mask'], items['obs_reference'] = self.preprocess_premise(observation)
# 		items['label'] = torch.tensor(self.data['label'][idx])
# 		items['pad_id'] = self.tokenizer.vocab['token2idx'][self.tokenizer.pad_token]

# 		return items

# 	def preprocess_hypothesis(self, hyp):
# 		hyp_tokens = self.tokenizer.tokenize(hyp)
# 		hyp_tokens.insert(0, self.tokenizer.start_token)
# 		hyp_tokens.append(self.tokenizer.end_token)
# 		hyp_ids = self.tokenizer.convert_tokens_to_ids(hyp_tokens)
# 		masks = [1]*len(hyp_ids)
# 		return torch.tensor(hyp_ids), torch.tensor(masks), hyp

# 	def preprocess_premise(self, obs):
# 		obs = (' ' + self.tokenizer.split_token + ' ').join(obs) # sentence </s> sentence 
# 		tokens = self.tokenizer.tokenize(obs)
# 		tokens.insert(0, self.tokenizer.start_token)
# 		tokens.append(self.tokenizer.end_token)
# 		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
# 		masks = [1]*len(tokens_id)
# 		return torch.tensor(tokens_id), torch.tensor(masks), obs

# def merge(sequences, pad_id):
# 	lengths = [len(l) for l in sequences]
# 	max_length = max(lengths)

# 	padded_batch = torch.full((len(sequences), max_length), pad_id).long()
# 	#pad to max_length
# 	for i, seq in enumerate(sequences):
# 		padded_batch[i, :len(seq)] = seq

# 	return padded_batch, torch.LongTensor(lengths)

# def alpha_collate_fn_base(batch):
# 	item={}
# 	for key in batch[0].keys():
# 		item[key] = [d[key] for d in batch] # [item_dic, item_idc ]

# 	pad_id = item['pad_id'][0]
# 	hyp1, hyp1_length = merge(item['hyp1'], pad_id)
# 	hyp1_mask, _ = merge(item['hyp1_mask'], pad_id)
# 	hyp2, hyp2_length = merge(item['hyp2'], pad_id)
# 	hyp2_mask, _ = merge(item['hyp2_mask'], pad_id)
# 	obs, obs_length = merge(item['obs'], pad_id)
# 	obs_mask, _ = merge(item['obs_mask'], pad_id)
# 	label = torch.stack(item['label']).float()

# 	d = {}
# 	d['hyp1'] = hyp1
# 	d['hyp1_length'] = hyp1_length
# 	d['hyp1_mask'] = hyp1_mask
# 	d['hyp1_reference'] = item['hyp1_reference']

# 	d['hyp2'] = hyp2
# 	d['hyp2_length'] = hyp2_length
# 	d['hyp2_mask'] = hyp2_mask
# 	d['hyp2_reference'] = item['hyp2_reference']

# 	d['obs'] = obs
# 	d['obs_length'] = obs_length
# 	d['obs_mask'] = obs_mask
# 	d['obs_reference'] = item['obs_reference']
# 	d['label'] = label

# 	return d


# def load_dataloader_base(dataset, test_dataset, val_dataset, batch_size, shuffle=True, drop_last = True, num_workers = 0):
# 	dataloader = DataLoader(dataset, 
# 		batch_size, 
# 		collate_fn = alpha_collate_fn_base, 
# 		shuffle=shuffle, 
# 		drop_last=drop_last,
# 		num_workers=num_workers )

# 	test_dataloader = DataLoader(test_dataset, 
# 		batch_size, 
# 		collate_fn = alpha_collate_fn_base, 
# 		shuffle=False, 
# 		drop_last=False,
# 		num_workers=num_workers )

# 	val_dataloader = None
# 	if val_dataset is not None:
# 		val_dataloader = DataLoader(val_dataset, 
# 			batch_size, 
# 			collate_fn = alpha_collate_fn_base, 
# 			shuffle=shuffle, 
# 			drop_last=False,
# 			num_workers=num_workers )
# 	return dataloader, test_dataloader, val_dataloader


# output_path = os.path.join(args.output_dir, args.output_name + '.h5')
# with h5py.File(output_path, 'w') as stories:
#     num_examples = len(dataset)
#     label_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
#     ob_dataset = stories.create_dataset('observation', (num_examples, args.max_ob_len), dtype= np.int64)
#     hyp1_dataset =stories.create_dataset('hypothesis1', (num_examples, args.max_hyp_len), dtype= np.int64)
#     hyp2_dataset =stories.create_dataset('hypothesis2', (num_examples, args.max_hyp_len), dtype= np.int64)
#     label_dataset =stories.create_dataset('label', (num_examples,),dtype= label_dtype)
    
#     '''
#     C number of datsets 
#     '''

#     for i, (sid, obs1, obs2, hyp1, hyp2, label) in enumerate(dataset):
#         ob_dataset[i] = obs1
#         hyp1_dataset[i] = hyp1
#         hyp2_dataset[i] = hyp2
#         label_dataset[i] = 0 if int(label) == 1 else 1  