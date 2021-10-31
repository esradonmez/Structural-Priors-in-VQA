from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg
from .lcgn import LCGN, LCGNConceptVocabulary, LCGNConceptVocabularyPD, LCGNConceptVocabularyRC, LCGNConceptVocabularyRCDI, LCGNConceptVocabularyRCDIS, LCGNConceptVocabularyTensor
from .input_unit import Encoder, EncoderConceptVocabulary, EncoderDecoderConceptVocabulary, EncoderConceptVocabularyTensor
from .output_unit import Classifier


class SingleHop(nn.Module):
    def __init__(self):
        super().__init__()
        if cfg.DIFF_SINGLE_HOP:
            if cfg.DIFF_SINGLE_HOP_TYPE == 'CONCAT':
                self.proj_q = ops.Linear(cfg.ENC_DIM + cfg.CPT_EMB_DIM, cfg.CTX_DIM)
            elif cfg.DIFF_SINGLE_HOP_TYPE == 'CMD':
                self.proj_q = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM)
        else:
            self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)

        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att


class LCGNnet(nn.Module):
    '''
    1. make this compatible with the concept vocabulary input 
        tensor input of size : num_objects X num_concepts X concept embedding size vector
    
    2. add question encoder that takes the question and maps it to vectors from the concept vocabulary 

    '''
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab-1, cfg.WRD_EMB_DIM)
        self.num_vocab = num_vocab
        self.num_choices = num_choices
        self.encoder = Encoder(embeddingsInit)
        self.lcgn = LCGN()
        self.single_hop = SingleHop()
        self.classifier = Classifier(num_choices)

    def forward(self, batch):
        batchSize = len(batch['image_feat_batch'])
        questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda()
        questionLengths = torch.from_numpy(
            batch['seq_length_batch'].astype(np.int64)).cuda()
        answerIndices = torch.from_numpy(
            batch['answer_label_batch'].astype(np.int64)).cuda()
        images = torch.from_numpy(
            batch['image_feat_batch'].astype(np.float32)).cuda()
        imagesObjectNum = torch.from_numpy(
            np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda()


        # LSTM
        '''
        compatible with concept question encoder?

        yes - EncoderConceptVocabulary still outputs questionCntxWords, vecQuestions
        '''

        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)
        
        '''
        Here do concept parrallel lcgn

        '''
        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)

        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        logits = self.classifier(x_att, vecQuestions)

        predictions, num_correct = self.add_pred_op(logits, answerIndices)
        loss = self.add_answer_loss_op(logits, answerIndices)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "loss": loss,
                "accuracy": float(num_correct * 1. / batchSize)}

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctNum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss


class LCGNwrapper():
    def __init__(self, num_vocab, num_choices):
        if cfg.CONCEPT_VOCABULARY:
            print('Using Concept Vocabulary')
            self.model  = LCGNnetConceptVocabulary(num_vocab, num_choices).cuda()
        else:
            self.model = LCGNnet(num_vocab, num_choices).cuda()

        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)
        self.lr = cfg.TRAIN.SOLVER.LR

        if cfg.USE_EMA:
            self.ema_param_dict = {
                name: p for name, p in self.model.named_parameters()
                if p.requires_grad}
            self.ema = ops.ExponentialMovingAverage(
                self.ema_param_dict, decay=cfg.EMA_DECAY_RATE)
            self.using_ema_params = False

    def train(self, training=True):
        self.model.train(training)
        if training:
            self.set_params_from_original()
        else:
            self.set_params_from_ema()

    def eval(self):
        self.train(False)

    def state_dict(self):
        # Generate state dict in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict() if cfg.USE_EMA else None
        }

        # restore original mode
        self.train(current_mode)

    def load_state_dict(self, state_dict):
        # Load parameters in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            print('Optimizer does not exist in checkpoint! '
                  'Loaded only model parameters.')

        if cfg.USE_EMA:
            if 'ema' in state_dict and state_dict['ema'] is not None:
                self.ema.load_state_dict(state_dict['ema'])
            else:
                print('cfg.USE_EMA is True, but EMA does not exist in '
                      'checkpoint! Using model params to initialize EMA.')
                self.ema.load_state_dict(
                    {k: p.data for k, p in self.ema_param_dict.items()})

        # restore original mode
        self.train(current_mode)

    def set_params_from_ema(self):
        if (not cfg.USE_EMA) or self.using_ema_params:
            return

        self.original_state_dict = deepcopy(self.model.state_dict())
        self.ema.set_params_from_ema(self.ema_param_dict)
        self.using_ema_params = True

    def set_params_from_original(self):
        if (not cfg.USE_EMA) or (not self.using_ema_params):
            return

        self.model.load_state_dict(self.original_state_dict)
        self.using_ema_params = False

    def run_batch(self, batch, train, lr=None):
        assert train == self.model.training
        assert (not train) or (lr is not None), 'lr must be set for training'

        if train:
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            batch_res = self.model.forward(batch)
            loss = batch_res['loss']
            loss.backward()
            if cfg.TRAIN.CLIP_GRADIENTS:
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
            self.optimizer.step()
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            with torch.no_grad():
                batch_res = self.model.forward(batch)

        return batch_res

class LCGNnetConceptVocabulary(nn.Module):
    '''
    1. make this compatible with the concept vocabulary input 
        tensor input of size : num_objects X num_concepts X concept embedding size vector
    
    2. add question encoder that takes the question and maps it to vectors from the concept vocabulary 

    '''
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab-1, cfg.WRD_EMB_DIM)

        if cfg.INIT_CPT_EMB_FROM_FILE:
            conceptsInit = np.load(cfg.CPT_EMB_INIT_FILE)
            # assert conceptsInit.shape == (cfg.NUM_CPT-1, cfg.CPT_EMB_DIM)
            # here randomly initialize one and add one 
            if cfg.CPT_TYPE == 'TENSOR':
                conceptSections = np.load(cfg.CPT_SECTIONS_INIT_FILE)
                cptTensorInit = conceptsInit
                conceptsInit = conceptsInit[:conceptSections[0]]
            conceptsInit = np.vstack((conceptsInit, np.random.rand(1, cfg.CPT_EMB_DIM)))
        else:
            #conceptsInit = np.random.rand(cfg.NUM_CPT+1, cfg.CPT_EMB_DIM)
            conceptsInit = np.random.randn(cfg.NUM_CPT, cfg.CPT_EMB_DIM) / cfg.NUM_CPT

            if cfg.CPT_TYPE == 'TENSOR':
                conceptSections = np.load(cfg.CPT_SECTIONS_INIT_FILE)
                cptTensorInit = np.random.randn(sum(CPT_SECTIONS_INIT_FILE), cfg.CPT_EMB_DIM) / cfg.NUM_CPT
                conceptsInit = cptTensorInit[:conceptSections[0]]
            
            conceptsInit = np.vstack((conceptsInit, np.random.rand(1, cfg.CPT_EMB_DIM)))

        # define the encoder using concept vocabulary 
        if  cfg.ENC_TYPE == 'enc':
            self.encoder = EncoderConceptVocabulary(embeddingsInit, conceptsInit)
        elif cfg.ENC_TYPE == 'enc-dec':
            self.encoder = EncoderDecoderConceptVocabulary(embeddingsInit)
        elif cfg.ENC_TYPE == 'enc-tensor': 
            conceptsInit = np.vstack((cptTensorInit[conceptSections[0]:], np.random.rand(1, cfg.CPT_EMB_DIM)))
            self.encoder= EncoderConceptVocabularyTensor(embeddingsInit, conceptsInit)
        else:
            self.encoder = EncoderConceptVocabulary(embeddingsInit, conceptsInit)

        self.num_vocab = num_vocab
        self.num_choices = num_choices
        #self.encoder = Encoder(embeddingsInit)
        if cfg.CPT_TYPE == 'PD':
            print('Only allowing interaction through a probabilty distribution')
            self.lcgn = LCGNConceptVocabularyPD()
        elif cfg.CPT_TYPE == 'RC':
            print('Allowing to refine concepts at each iteration')
            self.lcgn = LCGNConceptVocabularyRC()
        elif cfg.CPT_TYPE == 'RCDI':
            self.lcgn = LCGNConceptVocabularyRCDI(conceptsInit)
        elif cfg.CPT_TYPE == 'RCDIS':
            self.lcgn = LCGNConceptVocabularyRCDIS(conceptsInit)
        
        elif cfg.CPT_TYPE == 'TENSOR':
            self.lcgn = LCGNConceptVocabularyTensor(cptTensorInit, conceptSections)
        else:
            self.lcgn = LCGNConceptVocabulary()
        
        self.single_hop = SingleHop()
        self.classifier = Classifier(num_choices)

    def forward(self, batch):
        batchSize = len(batch['image_feat_batch'])
        questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda()
        questionLengths = torch.from_numpy(
            batch['seq_length_batch'].astype(np.int64)).cuda()
        answerIndices = torch.from_numpy(
            batch['answer_label_batch'].astype(np.int64)).cuda()
        images = torch.from_numpy(
            batch['image_feat_batch'].astype(np.float32)).cuda()
        imagesObjectNum = torch.from_numpy(
            np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda()

        questionCntxWords, vecQuestions, embeddingsConceptVar = self.encoder(
            questionIndices, questionLengths)
        
        '''
        Here do concept parrallel lcgn

        '''

        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum, 
            embeddingsConceptVar = embeddingsConceptVar)

        if cfg.CPT_TYPE in ['RCDI', 'RCDIS']:
            vecQuestions = vecQuestions[0]
        
        if cfg.DIFF_SINGLE_HOP:
            x_out, cmd = x_out
            if cfg.DIFF_SINGLE_HOP_TYPE == 'concat':
                vecQuestions = torch.cat([cmd, vecQuestions], dim=-1)
            elif cfg.DIFF_SINGLE_HOP_TYPE == 'cmd':
                vecQuestions = cmd
        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        logits = self.classifier(x_att, vecQuestions)

        predictions, num_correct = self.add_pred_op(logits, answerIndices)
        loss = self.add_answer_loss_op(logits, answerIndices)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "loss": loss,
                "accuracy": float(num_correct * 1. / batchSize)}

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctNum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss
