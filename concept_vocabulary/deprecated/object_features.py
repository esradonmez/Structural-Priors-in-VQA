import torch
import numpy as np

import os.path as osp
from glob import glob
from collections import defaultdict 
import text_processing

import json
import h5py

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
        
class SceneGraphFeatureLoader:
    def __init__(self, scene_graph_file, vocab_name_file, vocab_attr_file):
        '''
        vocab_name_file : contains all object names 
        vocab_attr_file : contains all attributes 
        '''
        print('Loading scene graph from %s' % scene_graph_file)
        with open(scene_graph_file) as f:
            self.SGs = json.load(f)
        print('Done')
        self.name_dict = text_processing.VocabDict(vocab_name_file)
        self.attr_dict = text_processing.VocabDict(vocab_attr_file)
        self.num_name = self.name_dict.num_vocab
        self.num_attr = self.attr_dict.num_vocab

    def load_feature_normalized_bbox(self, imageId):
        sg = self.SGs[imageId]
        num = len(sg['objects'])

        feature = np.zeros(
            (num, self.num_name+self.num_attr), np.float32)
        names = feature[:, :self.num_name]
        attrs = feature[:, self.num_name:]
        bbox = np.zeros((num, 4), np.float32)

        objIds = sorted(sg['objects'])[:num]
        boundingbox = {}
        for idx, objId in enumerate(objIds):
            obj = sg['objects'][objId]
            bbox[idx] = obj['x'], obj['y'], obj['w'], obj['h']
            boundingbox[objId] = bbox
            names[idx, self.name_dict.word2idx(obj['name'])] = 1.
            for a in obj['attributes']:
                attrs[idx, self.attr_dict.word2idx(a)] = 1.

        # xywh -> xyxy
        bbox[:, 2] += bbox[:, 0] - 1
        bbox[:, 3] += bbox[:, 1] - 1

        # normalize the bbox coordinates
        # TODO: Dow we need to normalize these?
        w, h = sg['width'], sg['height']
        normalized_bbox = bbox / [w, h, w, h]

        for objId, bb in zip(boundingbox.keys(), normalized_bbox):
            boundingbox[objId] = bb
        valid = get_valid(len(bbox), num)

        return normalized_bbox, boundingbox

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
        return bbox

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
        return normalized_bbox
    

def get_valid(total_num, valid_num):
    valid = np.zeros(total_num, np.bool)
    valid[:valid_num] = True
    return valid

class ObjectsFeatures:
    def __init__(self, feature_dir, scene_graph, vocab_name_file, vocab_attr_file):
        info_file = osp.join(feature_dir, 'gqa_objects_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)
        self.feature_dir = feature_dir
        self.scene_graph = scene_graph
        self.vocab_name_file = vocab_name_file
        self.vocab_attr_file = vocab_attr_file
    
    def calculate_iou(self, bbox1, bbox2):
        bb1 = dict()
        bb2 = dict()
        bb1['x1'] = bbox1[0]
        bb1['y1'] = bbox1[1]
        bb1['x2'] = bbox1[0] + bbox1[2]
        bb1['y2'] = bbox1[3] - bbox1[1]

        bb2['x1'] = bbox2[0]
        bb2['y1'] = bbox2[1]
        bb2['x2'] = bbox2[0] + bbox2[2]
        bb2['y2'] = bbox2[3] - bbox2[1]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box.
        # NOTE: We MUST ALWAYS add +1 to calculate area when working in
        # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
        # is the bottom right pixel. If we DON'T add +1, the result is wrong.
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
        bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
    
    def get_object_bboxes(self, imageId):
        sg_loader = SceneGraphFeatureLoader(self.scene_graph, self.vocab_name_file, self.vocab_attr_file)
        sg_bboxes, bb_dict = sg_loader.load_feature_normalized_bbox(imageId)
        #print("bboxes from scene graph for image 2386621:\n")
        #print(bb_scene_graph)

        obj_loader =ObjectsFeatureLoader(self.feature_dir)
        obj_bboxes = obj_loader.load_feature_normalized_bbox(imageId)
        #print("bboxes for image 2386621:\n")
        #bboxes = bboxes[:18]
        #print(bboxes)

        gold_bboxes = {}
        for i, bbox in enumerate(sg_bboxes):
            gold_bboxes[i] = bbox
        #print(gold_bboxes)
        
        extracted_bboxes = {}
        for i, bbox in enumerate(obj_bboxes):
            extracted_bboxes[i] = bbox
        #print(extracted_bboxes)

        # for every obj in in image
        # map obj -> bounding box idxes in object feature files
        items = {}
        for objId, (k1, bbox1) in zip(bb_dict.keys(), gold_bboxes.items()):
            matches = []
            for k2, bbox2 in extracted_bboxes.items():
                iou = self.calculate_iou(bbox1, bbox2)
                if iou > 0.5:
                    matches.append(k2)
            items[objId] = matches
        
        #print(items)
        #print(boundingbox)
        return items
    
    def map_objects_to_features(self, imageId):
        # for obj in scene graph, return list of features
        objects = self.get_object_bboxes(imageId)

        obj_loader = ObjectsFeatureLoader(self.feature_dir)

        features = obj_loader.load_feature(imageId)

        obj_features = {}
        for objId, indexes in objects.items():
            obj_features[objId] = list()
            for idx in indexes:
                feature = features[0][idx]
                obj_features[objId].append(feature)
        return obj_features


if __name__ == "__main__":
    spatial_path = "/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/data/spatial"
    scene_graph = "/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/data/sceneGraphs/train_sceneGraphs.json"
    vocab_name_file = "/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/AML-group-3/lcgn/exp_gqa/data/name_gqa.txt"
    vocab_attr_file = "/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/AML-group-3/lcgn/exp_gqa/data/attr_gqa.txt"
    object_path = "/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/data/objects"

    loader = ObjectsFeatures(object_path, scene_graph, vocab_name_file, vocab_attr_file)

    features = loader.map_objects_to_features('2386621')
    print(features)