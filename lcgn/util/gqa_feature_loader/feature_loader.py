import h5py
import json
import numpy as np
import os.path as osp
from glob import glob
from collections import defaultdict 

from util import text_processing


class SpatialFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_spatial_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)

        num_files = len(glob(osp.join(feature_dir, 'gqa_spatial_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_spatial_%d.h5' % n)
                    for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def __del__(self):
        for f in self.h5_files:
            f.close()

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx = info['file'], info['idx']
        return self.h5_files[file]['features'][idx]


class ObjectsFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_objects_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)

        num_files = len(glob(osp.join(feature_dir, 'gqa_objects_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_objects_%d.h5' % n)
                    for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def __del__(self):
        for f in self.h5_files:
            f.close()

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        feature = self.h5_files[file]['features'][idx]
        valid = get_valid(len(feature), num)
        return feature, valid

    def load_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return bbox, valid

    def load_feature_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return feature, bbox, valid

    def load_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return normalized_bbox, valid

    def load_feature_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return feature, normalized_bbox, valid


class SceneGraphFeatureLoader:
    def __init__(self, scene_graph_file, vocab_name_file, vocab_attr_file,
                 max_num):
        '''
        vocab_name_file : contains all object names 
        vocab_attr_file : contains all attributes 
        max_num : maximum number of objects to include 
        '''
        print('Loading scene graph from %s' % scene_graph_file)
        with open(scene_graph_file) as f:
            self.SGs = json.load(f)
        print('Done')
        self.name_dict = text_processing.VocabDict(vocab_name_file)
        self.attr_dict = text_processing.VocabDict(vocab_attr_file)
        self.num_name = self.name_dict.num_vocab
        self.num_attr = self.attr_dict.num_vocab
        self.max_num = max_num

    def load_feature_normalized_bbox(self, imageId):
        sg = self.SGs[imageId]
        num = len(sg['objects'])
        # if num > self.max_num:
        #     print('truncating %d objects to %d' % (num, self.max_num))

        feature = np.zeros(
            (self.max_num, self.num_name+self.num_attr), np.float32)
        names = feature[:, :self.num_name]
        attrs = feature[:, self.num_name:]
        bbox = np.zeros((self.max_num, 4), np.float32)

        objIds = sorted(sg['objects'])[:self.max_num]
        for idx, objId in enumerate(objIds):
            obj = sg['objects'][objId]
            bbox[idx] = obj['x'], obj['y'], obj['w'], obj['h']
            names[idx, self.name_dict.word2idx(obj['name'])] = 1.
            for a in obj['attributes']:
                attrs[idx, self.attr_dict.word2idx(a)] = 1.

        # xywh -> xyxy
        bbox[:, 2] += bbox[:, 0] - 1
        bbox[:, 3] += bbox[:, 1] - 1

        # normalize the bbox coordinates
        w, h = sg['width'], sg['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)

        return feature, normalized_bbox, valid

# #modified------------------------------------------------------------------

# class SceneGraphFeatureLoaderConceptVocabulary:
#     '''
#     modified from SceneGraphFeatureLoader class 
#     here the scene graph is loaded and mapped to the defined concept vocabulary 
#     '''
#     def __init__(self, 
#     scene_graph_file, 
#     concept_vocab_file, 
#     glove_name,  
#     max_num):
#         '''
#         we need to prepare three files as was done in lcgn 

#         1. concept vocablary here loads the concept embeddings and define the mapping - see text processing 
#         2. glove_name is to be supplied for the concept vocabulary object 
#         '''
#         print('Loading scene graph from %s' % scene_graph_file)
#         with open(scene_graph_file) as f:
#             self.SGs = json.load(f)
#         print('Done')
#         self.cncpt_dict = text_processing.ConceptDict(concept_vocab_file, glove_name)
#         self.num_cnpct = self.cncpt_dict.num_vocab
#         self.cncpt_emb_size = self.concpt_dict.cncpt_emb_size 
#         self.max_num = max_num

#     def load_feature_normalized_bbox(self, imageId):
#         '''
#         this takes the scene graph and prepares the features using the concept vocabulary
        
#         outputs
#         1. feature tensor of size: num_objects X num_concepts X concept embedding size vector
#         2. normalized bounding box information 
#         3. boudning box is valid or not  
#         '''
#         sg = self.SGs[imageId]
#         num = len(sg['objects'])
#         # if num > self.max_num:
#         #     print('truncating %d objects to %d' % (num, self.max_num))

#         feature = np.zeros(
#             (self.max_num, self.num_cnpct, self.cncpt_emb_size), np.float32)
#         bbox = np.zeros((self.max_num, 4), np.float32)

#         objIds = sorted(sg['objects'])[:self.max_num]
#         for idx, objId in enumerate(objIds):
#             obj = sg['objects'][objId]
#             bbox[idx] = obj['x'], obj['y'], obj['w'], obj['h']

#             # object name concept 
#             cncpt = self.cncpt_dict.attr2cncpt[obj['name']]
#             feature[idx, cncpt,:] =  self.cncpt_dict.cncpt_emb[cncpt][obj['name']]

#             # here use iterative averaging method to average the attributes to get concept embedding features 
#             N = np.zeros(len(self.num_cnpct))
#             for a in obj['attributes']:
#                 cncpt = self.cncpt_dict.attr2cncpt[a]
#                 cncpt_emb = self.cncpt_dict.cncpt_emb[cncpt][a]
#                 N[cncpt] += 1
#                 feature[idx, cncpt, :] += (1/N[cncpt])*(feature[idx, cncpt, :] - cncpt_emb)
#         # xywh -> xyxy
#         bbox[:, 2] += bbox[:, 0] - 1
#         bbox[:, 3] += bbox[:, 1] - 1

#         # normalize the bbox coordinates
#         w, h = sg['width'], sg['height']
#         normalized_bbox = bbox / [w, h, w, h]
#         valid = get_valid(len(bbox), num)

#         return feature, normalized_bbox, valid

# #modified------------------------------------------------------------------


def get_valid(total_num, valid_num):
    valid = np.zeros(total_num, np.bool)
    valid[:valid_num] = True
    return valid
