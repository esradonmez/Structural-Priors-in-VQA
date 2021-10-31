import json
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import glob
import h5py
'''
data loader 
- write a dataset loader that will compute the dataloader object 
'''

def prepare_concept_vocabulary(path='data/concepts.json',
                                object_path ='AML-group-3/lcgn/exp_gqa/data/name_gqa.txt'):

#for now assume concept vocabulary 
    concept_vocab = json.load(open(path, 'r')) # this needs to be concept_vocab[attr] = concept idx 

    #here fix the concept json into the form that we want 
    val2concept = {value: (int(ccIdx)+1, int(ccVIdx)) for ccIdx, val in concept_vocab.items() for value, (_, ccVIdx) in val.items()}

    # get object names for now like  this ... but fix later 
    object_names = [ name.strip() for name in open(object_path, 'r').readlines()]
    #object_rels = [ rel.strip() for rel in open('AML-group-3/lcgn/exp_gqa/data/rel_gqa.txt', 'r').readlines()]

    #add in object name at idx 0
    for idx, name in enumerate(object_names):
        val2concept[name] = (0, idx)

    concept2val = {}
    for  ccIdx, values  in concept_vocab.items():
        concept2val[int(ccIdx)+1] = list(values.keys())
    concept2val[0] =  len(object_names)

    return concept_vocab, val2concept, concept2val

class ConceptClassifierDataset(Dataset):

    def __init__(self, data_type,  iou_alpha=0.5):

        '''
        all object info are in gqa_objects_info.json

        dynamically calculates the label
        '''

        self.objects_info = json.load(open('../../data/objects/gqa_objects_info.json', 'r'))
        if data_type == "train":
            self.scene_graph_file = json.load(open('../../data/sceneGraphs/train_sceneGraphs.json','r'))
        if data_type == "val":
            self.scene_graph_file = json.load(open('../../data/sceneGraphs/val_sceneGraphs.json','r'))

        #load object feature files 
        num_files = len(glob.glob('../../data/objects/gqa_objects_*.h5'))
        h5_paths = [('../../data/objects/gqa_objects_%d.h5' % n)
                    for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

        # pre-calculate the number of objects and get imgId - but only for the train!
        self.object_ids  = [(imgId, objId) for imgId, img in self.objects_info.items() for objId in range(int(img['objectsNum'])) if imgId in self.scene_graph_file]
        self.alpha = iou_alpha

        #for now assume concept vocabulary 
        self.concept_vocab = json.load(open('../../data/concepts.json', 'r')) # this needs to be concept_vocab[attr] = concept idx 

        #here fix the concept json into the form that we want 
        self.val2concept = {value: (int(ccIdx)+1, int(ccVIdx)) for ccIdx, val in self.concept_vocab.items() for value, (_, ccVIdx) in val.items()}

        # get object names for now like  this ... but fix later 
        object_names = [ name.strip() for name in open('../../AML-group-3/lcgn/exp_gqa/data/name_gqa.txt', 'r').readlines()]
        #object_rels = [ rel.strip() for rel in open('AML-group-3/lcgn/exp_gqa/data/rel_gqa.txt', 'r').readlines()]

        #add in object name at idx 0
        for idx, name in enumerate(object_names):
            self.val2concept[name] = (0, idx)

        #for now do not include object relations
        # #add in rel name at last idx
        # for idx, rel in enumerate(object_rels):
        #     self.val2concept[rel] = (len(self.concept_vocab)+2, idx)

        self.concept2val = {}
        for  ccIdx, values  in self.concept_vocab.items():
            self.concept2val[int(ccIdx)+1] = len(values)
        self.concept2val[0] =  len(object_names)

        self.ccNum = len(self.concept2val)

    def __len__(self):
        return len(self.object_ids)
    
    def __getitem__(self, idx):
        '''
        here dynamically process data and give label 
        "width": 375, "objectsNum": 31, "idx": 7150, "height": 500, "file": 5
        '''
        #load object_id 
        imgId, objId = self.object_ids[idx]
        
        #load feature and bounding box
        feature, bbox = self.get_feature_normalized_bbox(imgId, objId)

        #load related Scene Graph objects  
        sGobjs = self.get_sG_normalized_bbox_cvals(imgId)
        
        #calcaulte the labels for the bbs 
        objLabel, ccLabel = self.get_obj_labels(bbox, sGobjs)

        
        return torch.tensor(feature), torch.tensor(objLabel), torch.tensor(ccLabel)
    
    def get_sG_normalized_bbox_cvals(self, imgId):
        sg = self.scene_graph_file[imgId]
        sg_objs = sg['objects']
        #get image width and height 
        img_w, img_h = sg['width'], sg['height']

        # get labels for concepts
        sGobjs = []
        for objNum, obj in sg_objs.items():
            #get name attributes 
            cval = []
            cval.append(obj['name'])
            cval.extend(obj['attributes'])
            #cval.extend(obj['relations']['name']) -- for now do not include relations

            #get normalized bounding box 
            bbox = self.normalize_bbox(obj, img_w, img_h)
            sGobjs.append((objNum, cval, bbox))
        return sGobjs
    
    def normalize_bbox(self, obj, img_w, img_h):
        w, h = obj['w'], obj['h']
        bbox = [obj['x'],obj['y'],obj['x']+w, obj['y']+h]
        normalized_bbox = [c/n for c, n in zip(bbox,[img_w, img_h, img_w, img_h])]
        return normalized_bbox
    
    def get_feature_normalized_bbox(self, imgId, objId):

        info = self.objects_info[imgId]

        #load object feature 
        obj_idx, obj_f = info['idx'], info['file']
        #load img feature 
        bbox = self.h5_files[obj_f]['bboxes'][obj_idx][objId]
        feature = self.h5_files[obj_f]['features'][obj_idx][objId]

        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
    
        return feature, normalized_bbox
    
    def get_obj_labels(self, bbx, sGobjs):

        objLabel = 0
        ccLabel = [0]*self.ccNum 
        for objNum, cval, objbbx in sGobjs:
            # number of values for each concept 
            if self.iou(bbx, objbbx) > self.alpha: 
                for val in cval:
                    # get  concept index 
                    ccIdx, ccVIdx = self.val2concept[val]
                    # get the concept value index 
                    ccLabel[int(ccIdx)] = int(ccVIdx)
                objLabel = 1 
                break
                
        return objLabel, ccLabel 

    def IoU(self, boxA, boxB):

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value

        assert(iou >= 0)
        assert(iou <= 1)
        return iou

def prepare_dataloader(
    batch_size=16,
    shuffle=True,
    num_workers = 16,
    drop_last =True, 
    iou_alpha = 0.5):
    train_dataset = ConceptClassifierDataset("train", iou_alpha=iou_alpha)
    val_dataset = ConceptClassifierDataset("val", iou_alpha=iou_alpha)
    train_dataloader = DataLoader(train_dataset, 
			batch_size, 
			shuffle=shuffle, 
			drop_last=drop_last,
			num_workers=num_workers )
    val_dataloader = DataLoader(val_dataset, 
			batch_size, 
			shuffle=shuffle, 
			drop_last=False,
			num_workers=num_workers )
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    
    print('loading the dataset ...')
    dataset = ConceptClassifierDataset("train")
    print(len(dataset))

    for i in range(10):
        idx = random.randint(0, len(dataset)-1)
        print(idx)
        sampled = dataset.__getitem__(idx)
        print(len(sampled[2]))