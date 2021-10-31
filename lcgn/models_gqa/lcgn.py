import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from entmax import sparsemax, entmax15

from . import ops as ops
from .config import cfg


class LCGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        elif cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())

        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length,
                entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                x_ctx_var_drop, entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        elif cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop

class LCGNConceptVocabulary(nn.Module):
    '''
    here instead of our model using an external concept classifier - we allow the model to learn this concept on its own 

    1. make sure that this concept vocabulary is shared by both question and lcgn models

    '''

    def __init__(self):
        super().__init__()

        assert(cfg.LCGN_CV_MODE  in ['concat', 'gate', 'none']) # concat allows you to also take original object features
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        
        self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        
        if cfg.LCGN_CV_MODE == 'concat':
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM // 2)
            self.initCKB = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM // 2)
        elif cfg.LCGN_CV_MODE == 'gate':
            self.initCKB = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM)
        else:
            self.initCKB = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM)
        
        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))
        self.nonCptVar = nn.Parameter(torch.randn(1, cfg.CPT_EMB_DIM),
                    requires_grad=(not cfg.CPT_EMB_FIXED))
        
        if cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                # nn.MaxPool2d((3,3), 1, padding=1),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                # nn.MaxPool2d((3,3), 1, padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())
    
            self.cpt_attn = ops.Linear(cfg.CTX_DIM, cfg.CPT_EMB_DIM)
            self.transImg = ops.Linear(cfg.CTX_DIM, cfg.CPT_EMB_DIM)
        
        elif cfg.NON_LINEAR_STEM:
            self.cpt_attn = ops.ClassProject(cfg.D_FEAT, cfg.CTX_DIM, 
            cfg.CPT_EMB_DIM, num_hidden_layers=cfg.NON_LINEAR_STEM_NUM_LAYERS, dropout=0.1)
            self.transImg = ops.Linear(cfg.D_FEAT, cfg.CPT_EMB_DIM)
        else:
            self.cpt_attn = ops.Linear(cfg.D_FEAT, cfg.CPT_EMB_DIM)
            self.transImg = ops.Linear(cfg.D_FEAT, cfg.CPT_EMB_DIM)

    def loc_ctx_init(self, images, embeddingsConceptVar):
        '''
        here classify each feature 

        CPT_EMB_DIM = M
        concept_embedding = C x M
        '''


        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
            #embeddingsConceptVar = F.normalize(embeddingsConceptVar, dim=-1)

        if  cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            images = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            images = self.cnn(images)
            images = images.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            images = torch.transpose(images, 1, 2)  # NC(HW) => N(HW)C
            # images = F.normalize(images, dim=-1)

        if cfg.LCGN_CV_MODE == 'gate':
            embeddingsConceptVar = torch.cat([embeddingsConceptVar, self.nonCptVar ], dim=0)



        embeddingsConceptVar = F.normalize(embeddingsConceptVar, dim=-1)
        img_queries = self.cpt_attn(images) 
        # here classify
        cpt_score = (torch.matmul(img_queries, torch.transpose(embeddingsConceptVar, 0, 1)) /
            np.sqrt(cfg.CPT_EMB_DIM)) # B x (N x M) X (M x C) 

        if cfg.SOFTMAX_TYPE == 'entmax':
            cpt_prob = entmax15(cpt_score, dim=-1)  # B x N x C
        elif cfg.SOFTMAX_TYPE == 'sparsemax':
            cpt_prob = sparsemax(cpt_score, dim=-1)  # B x N x C
        else:
            cpt_prob = F.softmax(cpt_score, dim=-1)  # B x N x C

        if cfg.LCGN_CV_MODE == 'gate':
            '''
            here we gate, hence we use the last non-concept's probabilty as the gating probabilty 

            '''
            # for gating we use the last embedding vector as the "non concept"
            cpt_embeddings = embeddingsConceptVar[:-1]
            non_cpt_embedding = embeddingsConceptVar[-1]
            p = cpt_prob[:, :, -1].unsqueeze(2)
            cpt_prob = cpt_prob[:, :, :-1]
            x_loc = (1-p)*torch.matmul(cpt_prob, cpt_embeddings) + p*self.transImg(images)
        else:
            #map to concepts 
            x_loc = torch.matmul(cpt_prob, embeddingsConceptVar) # B x ( N x C ) X (C x M)
    
        x_loc = self.initCKB(x_loc) 
        
        if cfg.LCGN_CV_MODE == 'concat':
            # if concat then we just concatenate 
            x_img_loc = self.initKB(images)
            x_loc = torch.cat([x_loc, x_img_loc], dim=-1)

        x_loc = self.x_loc_drop(x_loc)
        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        
        return x_loc, x_ctx, x_ctx_var_drop
    
    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length, embeddingsConceptVar, entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images, embeddingsConceptVar)
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                    q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                    x_ctx_var_drop, entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx
    
class LCGNConceptVocabularyPD(LCGNConceptVocabulary):
    '''
    here the restrictions are made so that the information is only allowed to pass between the L and V through a probaabilty distribution
    '''
    def __init__(self):
        super().__init__()

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        '''
        here we allow for the only interation through a probabilty distribution 
        - this way we get rid of the command being used in the value vectors of graph attention 
        '''
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) #* self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

class LCGNConceptVocabularyRC(LCGNConceptVocabulary):
    '''
    here we allow for the model to "recalcaulte" the concept vocabulary at each iteration 
    
    here in this model the contextual vectors from each iteration is mapped to the concept space 

    here we can think of different ways to accomplish this 
    1. allow local feature to be just the initial feature 
    2. allow 
    '''
    def __init__(self):
        super().__init__()
        self.build_concept_vocabulary()

    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length, embeddingsConceptVar, entity_num):
        x_loc, x_ctx, x_ctx_var_drop, embeddingsConceptVar = self.loc_ctx_init(images, embeddingsConceptVar)

        if cfg.RETURN_ATTN:
            attention_probs = []
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                    q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                    x_ctx_var_drop, entity_num, t, embeddingsConceptVar, attention_probs=attention_probs)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))

        if cfg.RETURN_ATTN:
            return x_out, attention_probs
        else:
            return x_out



    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t, embeddingsConceptVar, attention_probs):

    
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)

        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num, embeddingsConceptVar)
        
        if cfg.RETURN_ATTN:
            x_ctx = x_ctx, ctx_cpt_prob

        attention_probs.append(ctx_cpt_prob)
        return x_ctx

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num, embeddingsConceptVar):
        '''
        here we allow for the only interation through the 
        '''
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) #* self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))

        # here we allow for our newly formed contextual vector to be mapped to the context 
        if cfg.RETURN_ATTN:
            x_ctx_new, cpt_prob = self.concept_attention(x_ctx_new, embeddingsConceptVar)
            return x_ctx_new, cpt_prob
        else:
            x_ctx_new = self.concept_attention(x_ctx_new, embeddingsConceptVar)
            return x_ctx_new 
        


    def loc_ctx_init(self, images, embeddingsConceptVar):
        '''
        here we do not use the embeddingsConceptVar,
        we instead allow the model to use the concept embedding to think in terms of concepts at each message itereation 

        '''
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        embeddingsConceptVar = torch.cat([embeddingsConceptVar, self.nonCptVar ], dim=0)
        embeddingsConceptVar = F.normalize(embeddingsConceptVar, dim=-1)
        # here we c
        x_ctx = self.concept_attention(x_loc, embeddingsConceptVar)
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop, embeddingsConceptVar

    def build_loc_ctx_init(self):

        #assert(cfg.CPT_EMB_DIM == cfg.CTX_DIM)
        self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        self.cptAttn = ops.Linear(cfg.CTX_DIM, cfg.CPT_EMB_DIM)
        self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
        self.projCpt = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM)
        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    def concept_attention(self, x_loc, embeddingsConceptVar):
        queries = self.cptAttn(x_loc) 
        # here classify
        cpt_score = (torch.matmul(queries, torch.transpose(embeddingsConceptVar, 0, 1)) /
            np.sqrt(cfg.CPT_EMB_DIM)) # B x (N x M) X (M x C) 
        cpt_prob = F.softmax(cpt_score, dim=-1) # B x N x C

            # for gating we use the last embedding vector as the "non concept"
        cpt_embeddings = embeddingsConceptVar[:-1]
        p = cpt_prob[:, :, -1].unsqueeze(2)
        cpt_prob = cpt_prob[:, :, :-1]
        x_ctx = (1-p)*self.projCpt(torch.matmul(cpt_prob, cpt_embeddings)) + p*x_loc
        return x_ctx

    def build_concept_vocabulary(self):
        self.embeddingsConceptVar = nn.Parameter(
            torch.Tensor(cptInit), requires_grad=(not cfg.CPT_EMB_FIXED)) #do not require grad for now  
        self.nonCptVar = nn.Parameter(torch.randn(1, cfg.CPT_EMB_DIM),
                    requires_grad=(not cfg.CPT_EMB_FIXED))

class LCGNConceptVocabularyRCDI(LCGNConceptVocabulary):
    '''
    here we allow for the model to "recalcaulte" the concept vocabulary at each iteration 
    
    here in this model the contextual vectors from each iteration is mapped to the concept space 

    here we can think of different ways to accomplish this 
    1. allow local feature to be just the initial feature 

    dynamic input map 
        - use encoder decoder archiecture 
    '''
    def __init__(self, cptInit):
        super().__init__()
        self.build_concept_vocabulary(cptInit)
    def forward(self, images, q_encoding, lstm_outputs, q_length, batch_size, entity_num, embeddingsConceptVar):
        
        #here pepare for the input
        '''
        q_encoding = (h0, c0)
        '''
        embeddingsConceptVar = None
        cmd = self.cmd_init(batch_size)
        attention_probs = None
        if cfg.RETURN_ATTN:
            attention_probs = []
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx, q_encoding, cmd = self.run_message_passing_iter(
                    q_encoding, x_loc, x_ctx,
                    x_ctx_var_drop, entity_num, lstm_outputs, q_length, cmd, attention_probs, t)
        
        if cfg.ONLY_CPT_OUT:
            x_out = self.combine_kb(x_ctx)
        else:
            x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        
        if cfg.DIFF_SINGLE_HOP:
            cmd, _ = self.extract_textual_command(cmd, q_encoding, lstm_outputs, q_length)
            x_out = (x_out, cmd)
    
        if cfg.RETURN_ATTN:
            return x_out, attention_probs
        else:
            return x_out

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

        if cfg.ONLY_CPT_OUT:
            self.combine_kb = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        else:
            self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def run_message_passing_iter(
            self, q_encoding, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, lstm_outputs, q_length, cmd, attention_probs, t):
        
        cmd, q_encoding = self.extract_textual_command(
             cmd, q_encoding, lstm_outputs, q_length)

        if cfg.RETURN_ATTN:
             cmd, cmd_cpt_prob = cmd

        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num, t)
        
        if cfg.RETURN_ATTN:
            x_ctx, x_ctx_cpt_prob = x_ctx
            attention_probs.append((cmd_cpt_prob, x_ctx_cpt_prob))
            
        return x_ctx, q_encoding, cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num, t):
        '''
        here we allow for the only interation through the 
        '''
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        # here apply layernorm
        if cfg.ADD_LAYER_NORM:
            x_ctx = ops.activations[cfg.CPT_NON_LINEAR_TYPE](x_ctx)
            layernorm = getattr(self, "layernorm%d" % t)
            x_joint = layernorm(x_joint)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))

        # here we allow for our newly formed contextual vector to be mapped to the context 
        if cfg.RETURN_ATTN:
            x_ctx_new, cpt_prob = self.concept_attention(x_ctx_new)
            return x_ctx_new, cpt_prob
        else:
            x_ctx_new = self.concept_attention(x_ctx_new)
            return x_ctx_new 


    def loc_ctx_init(self, images):
        '''
        here we do not use the embeddingsConceptVar,
        we instead allow the model to use the concept embedding to think in terms of concepts at each message itereation 

        '''
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        # here we c
        x_ctx = self.concept_attention(x_loc)
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))

        return x_loc, x_ctx, x_ctx_var_drop
    
    def cmd_init(self, batch_size):
        cmd = self.initCmd.expand(batch_size, -1)
        return cmd

    def build_loc_ctx_init(self):
        #assert(cfg.CPT_EMB_DIM == cfg.CTX_DIM)
        self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
        self.projCptEmb = ops.Linear(cfg.CPT_EMB_DIM,  cfg.CTX_DIM)
        self.projCptCmd = ops.Linear(cfg.CPT_EMB_DIM,  cfg.CMD_DIM)    

        if cfg.ADD_LAYER_NORM:
            for t in range(cfg.MSG_ITER_NUM):
                layernorm = nn.LayerNorm(cfg.CTX_DIM*3)
                setattr(self, "layernorm%d" % t, layernorm)


    def concept_attention(self, x_loc):

        if cfg.CPT_ATTN_GATE:
            embeddingsConceptVar = torch.cat([self.embeddingsConceptVar[:-1], self.nonCptVar ], dim=0)
        else:
            embeddingsConceptVar = self.embeddingsConceptVar

        queries = self.cptAttnImg(x_loc) 
        
        embeddingsConceptVar = F.normalize(embeddingsConceptVar, dim=-1)
        # here classify
        cpt_score = (torch.matmul(queries, torch.transpose(self.embeddingsConceptVar, 0, 1)) /
            np.sqrt(cfg.CPT_EMB_DIM)) # B x (N x M) X (M x C) 

        if cfg.SOFTMAX_TYPE == 'entmax':
            cpt_prob = entmax15(cpt_score, dim=-1) # B x N x C
        elif cfg.SOFTMAX_TYPE == 'sparsemax':
            cpt_prob = sparsemax(cpt_score, dim=-1) # B x N x C
        else:
            cpt_prob = F.softmax(cpt_score, dim=-1) # B x N x C


        if cfg.CPT_ATTN_GATE:
            x_loc = self.transImg(x_loc)
            # for gating we use the last embedding vector as the "non concept"
            cpt_embeddings = embeddingsConceptVar[:-1]
            p = cpt_prob[:, :, -1].unsqueeze(2)
            cpt_probs = cpt_prob[:, :, :-1]
            x_ctx = (1-p)*self.projCptEmb(torch.matmul(cpt_probs, cpt_embeddings)) + p*x_loc
        else:
            x_ctx = self.projCptEmb(torch.matmul(cpt_prob, embeddingsConceptVar))
        
        # if cfg.CPT_NON_LINEAR:
        #     x_ctx = ops.activations[cfg.CPT_NON_LINEAR_TYPE](x_ctx)

        if cfg.RETURN_ATTN:
            return x_ctx , cpt_prob.detach()
        return x_ctx

    def extract_textual_command(self, prev_cmd, q_encoding, lstm_outputs, q_length):
        '''
        at iteration 0 : we initialize the qInput layer 
            (h0, c0) - here we take (h0, c0) from the question encoding layer 

        at each iteration t: 
            we use the hidden vectors to attend on the concept vectors 

        here q-length - is because the question lengths are different 

        here q_encoding is (h0, c0)
        '''
        # here instead, attend auto-regressively - here the input is the previous timestep output 
        h1, c1 = self.qDecoder(torch.zeros(prev_cmd.size(0),cfg.CPT_EMB_DIM).cuda(), q_encoding)
        # here we fixzed this from 
        #h1, c1 = self.qDecoder(prev_cmd.detach(), q_encoding)

        # here do attention to create the output vector that will be used for concept attention 
        raw_att = self.cmd_inter2logits(
        h1[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        cmd = torch.cat([h1, cmd], dim=-1)

        # here do concept attention 
        embeddingsConceptVar = self.embeddingsConceptVar # C x E
        cmd_queries = self.cptAttnCmd(cmd)  # B x E
        cmd = self.transCmd(cmd)  # B x H

        cpt_score = (torch.matmul(cmd_queries, torch.transpose(self.embeddingsConceptVar, 0, 1)) /
            np.sqrt(cfg.CPT_EMB_DIM)) # B x (N x M) X (M x C) 

        if cfg.SOFTMAX_TYPE == 'entmax':
            cpt_prob = entmax15(cpt_score, dim=-1) # B x N x C
        elif cfg.SOFTMAX_TYPE == 'sparsemax':
            cpt_prob = sparsemax(cpt_score, dim=-1) # B x N x C
        else:
            cpt_prob = F.softmax(cpt_score, dim=-1) # B x N x C

        if cfg.CPT_ATTN_GATE:
            cpt_embeddings = embeddingsConceptVar[:-1]
            p = cpt_prob[:,-1].unsqueeze(1)
            cpt_probs = cpt_prob[:, :-1]
            cmd = (1-p)*self.projCptCmd(torch.matmul(cpt_probs, cpt_embeddings)) + p*cmd
        
        else:
            cmd = self.projCptCmd(torch.matmul(cpt_prob, embeddingsConceptVar))
        
        # if cfg.CPT_NON_LINEAR:
        #     cmd = ops.activations[cfg.CPT_NON_LINEAR_TYPE](cmd)

        if cfg.RETURN_ATTN:
            return (cmd, cpt_prob.detach()) , (h1,c1)
        else:
            return cmd, (h1,c1)

    def build_extract_textual_command(self):
        self.qDecoder = nn.LSTMCell(
            input_size=cfg.CPT_EMB_DIM, hidden_size=cfg.CMD_DIM) # x2 because we are using BiLSTM encoder 
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)
        self.initCmd = nn.Parameter(torch.zeros(1, cfg.CPT_EMB_DIM), requires_grad=False)
        
    
    def build_concept_vocabulary(self, cptInit):
        self.nonCptVar = nn.Parameter(torch.randn(1, cfg.CPT_EMB_DIM),
                    requires_grad=(not cfg.CPT_EMB_FIXED))
        
        if cfg.CPT_NON_LINEAR_PROJECTION:
            self.cptAttnImg = ops.ClassProject(cfg.CTX_DIM, cfg.CTX_DIM, cfg.CPT_EMB_DIM)
            self.cptAttnCmd = ops.ClassProject(cfg.CMD_DIM*2, cfg.CMD_DIM, cfg.CPT_EMB_DIM)
        else:
            self.cptAttnImg = ops.Linear(cfg.CTX_DIM, cfg.CPT_EMB_DIM)
            self.cptAttnCmd = ops.Linear(cfg.CMD_DIM*2, cfg.CPT_EMB_DIM)

        if cfg.CPT_ATTN_GATE:
            self.transImg = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.transCmd = ops.Linear(cfg.CMD_DIM*2, cfg.CTX_DIM)
        self.embeddingsConceptVar = nn.Parameter(
            torch.Tensor(cptInit), requires_grad=(not cfg.CPT_EMB_FIXED)) #do not require grad for now  


class LCGNConceptVocabularyRCDIS(LCGNConceptVocabularyRCDI):
    '''
    only thing that is different here than RCDI is that 
    we allow for different cmd 
    '''
    def __init__(self, cptInit):
        super().__init__(cptInit)

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        '''
        here we allow for the only interation through the 
        '''
        
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)

        act_fun_1 = ops.activations[cfg.CMD_INPUT_ACT]
        act_fun_2 = ops.activations[cfg.CMD_INPUT_ACT]

        cmd_keys = self.proj_keys_2(act_fun_1(self.proj_keys_1(cmd)[:, None, :]))
        cmd_vals = self.proj_vals_2(act_fun_2(self.proj_vals_1(cmd)[:, None, :]))
        keys = self.keys(x_joint) * cmd_keys
        vals = self.vals(x_joint) * cmd_vals
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))

        # here we allow for our newly formed contextual vector to be mapped to the context 
        x_ctx_new = self.concept_attention(x_ctx_new)
        return x_ctx_new

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys_1 = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM //2)
        self.proj_vals_1 = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM //2)
        self.proj_keys_2 = ops.Linear(cfg.CTX_DIM//2, cfg.CTX_DIM)
        self.proj_vals_2 = ops.Linear(cfg.CTX_DIM//2, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)


class LCGNConceptVocabularyTensor(nn.Module):
    '''
    here instead of our model using an external concept classifier - we allow the model to learn this concept on its own 

    1. make sure that this concept vocabulary is shared by both question and lcgn models

    '''

    def __init__(self, cptTensorInit, conceptSections):
        super().__init__()


        self.conceptSections = tuple(conceptSections[1:])
        self.cptTensorInit = nn.Parameter(
            torch.Tensor(cptTensorInit[conceptSections[0]:]), requires_grad=(not cfg.CPT_EMB_FIXED))
        # self.embeddingsConceptVar = nn.Parameter(
        #     torch.Tensor(cptTensorInit[:conceptSections[0]]), requires_grad=(not cfg.CPT_EMB_FIXED))
        
        # concat allows you to also take original object features
        assert(cfg.LCGN_CV_MODE in ['concat', 'gate', 'none'])
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))
        self.initCKB = ops.Linear(cfg.CPT_EMB_DIM, cfg.CTX_DIM)

        # assert(cfg.STEM_CNN != cfg.NON_LINEAR_STEM)
        if cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                # nn.MaxPool2d((3,3), 1, padding=1),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.STEM_CNN_DIM,
                         (3, 3), padding=1),
                # nn.MaxPool2d((3,3), 1, padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())
    
            self.transImg = ops.Linear(cfg.CTX_DIM, cfg.CPT_EMB_DIM)
        elif cfg.NON_LINEAR_STEM:
            self.transImg = ops.ClassProject(cfg.D_FEAT, cfg.CTX_DIM, 
            cfg.CPT_EMB_DIM, num_hidden_layers=cfg.NON_LINEAR_STEM_NUM_LAYERS, dropout=0.1)
        else:
            self.transImg = ops.Linear(cfg.D_FEAT, cfg.CPT_EMB_DIM)

    def loc_ctx_init(self, images, embeddingsConceptVar):
        '''
        here classify each feature 

        CPT_EMB_DIM = M
        concept_embedding = C x M
        '''

        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
            #embeddingsConceptVar = F.normalize(embeddingsConceptVar, dim=-1)

        if cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            images = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            images = self.cnn(images)
            images = images.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            images = torch.transpose(images, 1, 2)  # NC(HW) => N(HW)C
            # images = F.normalize(images, dim=-1)

        x_loc = self.transImg(images)
        x_cpt = self.concept_attention(x_loc, embeddingsConceptVar)
        x_loc = self.initCKB(x_cpt + x_loc) 
        x_loc = self.x_loc_drop(x_loc)
        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))

        return x_loc, x_ctx, x_ctx_var_drop

    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length, embeddingsConceptVar, entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(
            images, embeddingsConceptVar)
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                x_ctx_var_drop, entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
            q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx
    
    def concept_attention(self, images, embeddingsConceptVar):
        '''
        supposes that we have C number of concepts and N number of names/attributes  

        then W = word embedding for the names and attributes 

        x = B x L x H
        W = M x H

        If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes in dim according to split_size_or_sections.
        
        W is shaped this way

        W[:C] = " mean concept vectors" 
        W[]
        '''

        logits = F.linear(images, self.cptTensorInit)  # B x L x H X M x H
        #now using the index we need to split this into n number of objects 
        # suppose that first index of the 
        concept_logits = torch.split(logits,self.conceptSections,dim=-1)
        conceptVectors = []
        conceptVs = torch.split(self.cptTensorInit.transpose(0,1), self.conceptSections ,dim=1)

        # embeddingsConceptVar = torch.cat([V.mean(dim=0,keepdim=True)for V in conceptVs], dim=0)
        conceptProblogits = F.linear(images, embeddingsConceptVar)
        conceptProbs = torch.sigmoid(
            conceptProblogits).unsqueeze(2)  # B x L x 1 x C

        for concept_logit, V in zip(concept_logits,conceptVs):
            concept_prob = F.softmax(concept_logit, dim=-1)
            concept_vector = F.linear(concept_prob,V) # B x L x M_ci x M_ci x H
            conceptVectors.append(concept_vector.unsqueeze(2))
        
        # print(conceptProbs[0,0, 0,:])
        B, L , _, C = conceptProbs.size() # B, L, 1, C x B, L, C, H
        conceptVector = torch.cat(conceptVectors,dim=2) # B x L x C x H
        conceptVector = torch.bmm(conceptProbs.view(
            B*L, -1, C), conceptVector.view(B*L, C, -1)) # B*L x 1 x H
        conceptVector = conceptVector.view(B, L, -1)
        return conceptVector