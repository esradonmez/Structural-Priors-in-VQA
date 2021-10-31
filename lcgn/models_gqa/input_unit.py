import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .config import cfg
from . import ops as ops

'''
 we will modify this code to allow for shared concept vocabulary 

'''


class Encoder(nn.Module):
    def __init__(self, embInit):
        super().__init__()
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.question_drop = nn.Dropout(1 - cfg.qDropout)

    def forward(self, qIndices, questionLengths):
        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()
        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings)
        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)
        vecQuestions = self.question_drop(vecQuestions)

        return questionCntxWords, vecQuestions

# out modification here-------------------------------------------- 
class EncoderConceptVocabulary(nn.Module):

    '''
    Here we do not fix the concept vocabulary 
    '''

    def __init__(self,embInit, cptInit):
        super().__init__()

        # embeddings for the tokens 
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        self.embeddingsConceptVar = nn.Parameter(
            torch.Tensor(cptInit), requires_grad=(not cfg.CPT_EMB_FIXED)) #do not require grad for now  
        
        self.simlinear0 = nn.Linear(cfg.CPT_EMB_DIM, cfg.WRD_EMB_DIM)
        nn.init.eye_(self.simlinear0.weight)
        self.linear0 = ops.Linear(cfg.CPT_EMB_DIM, cfg.WRD_EMB_DIM)
        
        #assert(cfg.CPT_EMB_DIM == cfg.WRD_EMB_DIM)
        #now this comes after mapping 
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.question_drop = nn.Dropout(1 - cfg.qDropout)
    
    def forward(self, qIndices, questionLengths):

        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()

        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings)

        # here we map to concepts
        #if before: # for now we just map to the concept space first 
        embeddingsConceptVar = self.embeddingsConceptVar.cuda()
        cpt_score = (torch.matmul(questions, torch.transpose(self.simlinear0(embeddingsConceptVar), 0, 1)) /
            np.sqrt(cfg.CPT_EMB_DIM)) # B x (N x M) X (M x C) 
        cpt_prob = F.softmax(cpt_score, dim=-1) # B x N x C
        cpt_embeddings = embeddingsConceptVar[:-1]
        non_cpt_embedding = embeddingsConceptVar[-1]
        p = cpt_prob[:, :, -1].unsqueeze(2)
        cpt_prob = cpt_prob[:, :, :-1]
        questions = (1-p)*self.linear0(torch.matmul(cpt_prob, cpt_embeddings)) + p*questions

        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)

        #if not before: # if not before we map the last hidden states to the concept embeddings 

        vecQuestions = self.question_drop(vecQuestions)

        return questionCntxWords, vecQuestions, embeddingsConceptVar[:-1, :] #[:-1,:]

class EncoderConceptVocabularyTensor(nn.Module):

    '''
    Here we do not fix the concept vocabulary  - but tensor 
    '''

    def __init__(self,embInit, cptInit):
        super().__init__()

        # embeddings for the tokens 
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        self.embeddingsConceptVar = nn.Parameter(
            torch.Tensor(cptInit), requires_grad=(not cfg.CPT_EMB_FIXED)) #do not require grad for now  
        
        self.simlinear0 = nn.Linear(cfg.WRD_EMB_DIM, cfg.CPT_EMB_DIM,  bias=False)
        nn.init.eye_(self.simlinear0.weight)

        self.linear0 = ops.Linear(cfg.CPT_EMB_DIM, cfg.WRD_EMB_DIM) # for chagning back to the word dim 
        
        #assert(cfg.CPT_EMB_DIM == cfg.WRD_EMB_DIM)
        #now this comes after mapping 
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.question_drop = nn.Dropout(1 - cfg.qDropout)
    
    def forward(self, qIndices, questionLengths):

        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()

        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings) # B x L X H

        # here we map to concepts
        #if before: # for now we just map to the concept space first 
        embeddingsConceptVar = self.embeddingsConceptVar.cuda() # M x C
        embeddingsConceptVar = embeddingsConceptVar
        simW = self.simlinear0(questions) # B x L x M 

        cpt_score = F.linear(simW, embeddingsConceptVar) # B x L x C
        cpt_prob = F.softmax(cpt_score, dim=-1) # B x N x C
        cpt_embeddings = embeddingsConceptVar[:-1] # M x C-1
        p = cpt_prob[:, :, -1].unsqueeze(2) # B x L x C
        cpt_prob = cpt_prob[:, :, :-1] # B x L x C-1

        questions = (1-p)*self.linear0(F.linear(cpt_prob, cpt_embeddings.transpose(0,1))) + p*questions
        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)

        #if not before: # if not before we map the last hidden states to the concept embeddings 

        vecQuestions = self.question_drop(vecQuestions)

        return questionCntxWords, vecQuestions, embeddingsConceptVar[:, :-1] #[:-1,:]

class EncoderDecoderConceptVocabulary(nn.Module):

    '''
    Here we do not fix the concept vocabulary 
    '''

    def __init__(self,embInit):
        super().__init__()

        # embeddings for the tokens 
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        
        #assert(cfg.CPT_EMB_DIM == cfg.WRD_EMB_DIM)
        #now this comes after mapping 
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTMed()
    
    def forward(self, qIndices, questionLengths):

        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()

        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings)
        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions  = self.rnn0(questions, questionLengths)

        return questionCntxWords, vecQuestions , None

# class EncoderConceptVocabularyNSM(nn.Module):
#     '''
#     This maps the input into

#     1. add CPT_EMB_DIM to config file 
#     2. assume that CPT_EMB_DIM and WRD_EMB_DIM are the same 

#     here what we do is we have a concept vocabulary 
#     what we do is we map the embeddings to the concept space 
#      - if its not close to any concepts then we just average the concept embeddings  

#      here the IMPORTANT DIFFERENCE IS THAT 
#         1. CONCEPT VOCABULARY EMBEDDING MATRIX DOES NOT TRAIN!
#     '''
#     def __init__(self, embInit, embeddingsConceptVar):
#         super().__init__()

#         # embeddings for the tokens 
#         self.embeddingsVar = nn.Parameter(
#             torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))

#         '''
#         concept init vector should be prepared

#         '''
#         # embeddings for the 
#         self.embeddingsConceptVar = nn.Parameter(
#             torch.Tensor(conceptInit), requires_grad=(False) #do not require grad for now  
#             )
        
#         #for now initilaize using zeros   - this is learned
#         self.embeddingNonConceptVar = nn.Parameter(
#             torch.randn((1, cfg.CPT_EMB_DIM)), requires_grad = (True)
#         )

#         assert(cfg.CPT_EMB_DIM == cfg.WRD_EMB_DIM) # for now assume that they are the same 
#         self.simlinear0 = nn.Linear(cfg.CPT_EMB_DIM ,cfg.WRD_EMB_DIM) # for similarity 
#         # initialize with identity matrix - as specified in NSM
#         torch.nn.init.eye_(self.simlinear0.weight)

#         #now this comes after mapping 
#         self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
#         self.rnn0 = BiLSTM()
#         self.question_drop = nn.Dropout(1 - cfg.qDropout)
    
#     def forward(self, qIndices, questionLengths):
#         # Word embedding
#         embeddingsVar = self.embeddingsVar.cuda()
#         embeddings = torch.cat(
#             [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
#             dim=0)
#         questions = F.embedding(qIndices, embeddings) # B x L x H

#         #map to concepts - caculate attention scores -concepts 
#         concepts = torch.cat([self.embeddingsConceptVar, self.embeddingNonConceptVar], dim=0) # C+1
#         concepts = self.simlinear0(concepts) # C+1 x H
#         atnweights = questions.matmul(concepts.transpose(0,1)) # B x L x C+1

#         #map to concepts - mapping - for now c and H are the same 
#         p = atnweights[:,:,-1] # B x L
#         p_bar = atnweights[:,:,:-1] # B x L x C
#         questions = questions*p +  torch.bmm(p_bar, self.embeddingsConceptVar[:,:-1]).sum(dim=-1) # C x c , taken from NSM

#         #quesiton encoding 
#         questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)
#         vecQuestions = self.question_drop(vecQuestions)

#         return questionCntxWords, vecQuestions
# # out modification here-------------------------------------------- 
        
class BiLSTM(nn.Module):
    def __init__(self, forget_gate_bias=1.):
        super().__init__()
        self.bilstm = torch.nn.LSTM(
            input_size=cfg.WRD_EMB_DIM, hidden_size=cfg.ENC_DIM // 2,
            num_layers=cfg.INPUT_NUM_LAYERS, batch_first=True, bidirectional=True)

        d = cfg.ENC_DIM // 2

        # initialize LSTM weights (to be consistent with TensorFlow)
        fan_avg = (d*4 + (d+cfg.WRD_EMB_DIM)) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

        # initialize LSTM forget gate bias (to be consistent with TensorFlow)
        self.bilstm.bias_ih_l0.data[...] = 0.
        self.bilstm.bias_ih_l0.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0.data[...] = 0.
        self.bilstm.bias_hh_l0.requires_grad = False
        self.bilstm.bias_ih_l0_reverse.data[...] = 0.
        self.bilstm.bias_ih_l0_reverse.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0_reverse.data[...] = 0.
        self.bilstm.bias_hh_l0_reverse.requires_grad = False

    def forward(self, questions, questionLengths):
        # sort samples according to question length (descending)
        sorted_lengths, indices = torch.sort(questionLengths, descending=True)
        sorted_questions = questions[indices]
        _, desorted_indices = torch.sort(indices, descending=False)

        # pack questions for LSTM forwarding
        packed_questions = nn.utils.rnn.pack_padded_sequence(
            sorted_questions, sorted_lengths.cpu(), batch_first=True)
        packed_output, (sorted_h_n, _) = self.bilstm(packed_questions)

        if cfg.INPUT_NUM_LAYERS > 1:
            sorted_h_n = sorted_h_n.view(2, cfg.INPUT_NUM_LAYERS, -1 ,cfg.ENC_DIM // 2)[:, -1, :, :]
        sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=questions.size(1))
        sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape(
            questions.size(0), -1)

        # sort back to the original sample order
        output = sorted_output[desorted_indices]
        h_n = sorted_h_n[desorted_indices]

        return output, h_n


# class BiLSTMed(nn.Module):
#     '''

#     for encoder decoder archiecture - we have to prepare the cell 
#     '''
#     def __init__(self, forget_gate_bias=1.):
#         super().__init__()
#         self.bilstm = torch.nn.LSTM(
#             input_size=cfg.WRD_EMB_DIM, hidden_size=cfg.ENC_DIM // 2,
#             num_layers=cfg.INPUT_NUM_LAYERS, batch_first=True, bidirectional=True)

#         d = cfg.ENC_DIM // 2

#         # initialize LSTM weights (to be consistent with TensorFlow)
#         fan_avg = (d*4 + (d+cfg.WRD_EMB_DIM)) / 2.
#         bound = np.sqrt(3. / fan_avg)
#         nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
#         nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
#         nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
#         nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

#         # initialize LSTM forget gate bias (to be consistent with TensorFlow)
#         self.bilstm.bias_ih_l0.data[...] = 0.
#         self.bilstm.bias_ih_l0.data[d:2*d] = forget_gate_bias
#         self.bilstm.bias_hh_l0.data[...] = 0.
#         self.bilstm.bias_hh_l0.requires_grad = False
#         self.bilstm.bias_ih_l0_reverse.data[...] = 0.
#         self.bilstm.bias_ih_l0_reverse.data[d:2*d] = forget_gate_bias
#         self.bilstm.bias_hh_l0_reverse.data[...] = 0.
#         self.bilstm.bias_hh_l0_reverse.requires_grad = False

#     def forward(self, questions, questionLengths):
#         # sort samples according to question length (descending)
#         sorted_lengths, indices = torch.sort(questionLengths, descending=True)
#         sorted_questions = questions[indices]
#         _, desorted_indices = torch.sort(indices, descending=False)

#         # pack questions for LSTM forwarding
#         packed_questions = nn.utils.rnn.pack_padded_sequence(
#             sorted_questions, sorted_lengths.cpu(), batch_first=True)
#         packed_output, (sorted_h_n, sorted_c_n) = self.bilstm(packed_questions)
#         sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
#             packed_output, batch_first=True, total_length=questions.size(1))
#         sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape(
#             questions.size(0), -1)
#         sorted_c_n = torch.transpose(sorted_c_n, 1, 0).reshape(
#             questions.size(0), -1)

#         # sort back to the original sample order
#         output = sorted_output[desorted_indices]
#         h_n = sorted_h_n[desorted_indices]
#         c_n = sorted_c_n[desorted_indices]

#         return output, (h_n, c_n)

if __name__ == '__main__':
    pass
