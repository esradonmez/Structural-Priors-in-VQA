# Concept Space Prior in Language and Vision models

Sungjun Han, Esra Dönmez

Project report: [Here](https://github.tik.uni-stuttgart.de/PascalTilli/AML-group-3/blob/master/report.pdf)

## Abstract

Problems at the intersection of language and vision are of significant importance, both as challenging research questions as well as industrial applications. Models based on transfer learning from a model pretrained using a large amount of unannotated multi-modal data is one of the most successful approach in the field of language and vision in recent times. Another approach tangent with the pretrain-transfer approach that focuses on using the appropriate structural priors without any pretraining data has also shown to be competitive. One model of this family called Neural State Machine used a prior that we refer to as the concept space prior to map both modalities to the same abstract semantic space for reasoning. This prior draws inspiration from the way humans think. In this work, we study the importance of the concept space prior in language and vision models. We do this by equipping a similar model called Language-conditioned Graph Network with the prior. We propose three different ways of modifying the model: Local-VCS, Dynamic-VCS, and MCS. We show that the priored LCGN is able to do as well as the original model and examine the results between the proposed models to elucidate the necessary mechanisms in mapping the modalities to the abstract concept space that are needed to support the prior.

## Code structure 
We directly add and modify the publicly released pytorch version of LCGN : https://github.com/ronghanghu/lcgn. Hence for more detailed information on the code-structure, we refer to the readers to the above link for LCGN. However, we will try to provide as much information on the code structure as we can here.

In our work, we propose three models: local-VCS, Dynamic-VCS, and MCS. 

The code structure is as follows :

1. concept_vocabulary : this folder holds all the python scripts used for the concept vocabulary along with other jupyter notebooks used for analysis on the models and the GQA dataset. 
2. lcgn : this folder holds all the code for LCGN and the modified code for the three models 

We add the suffix "\*" in front of the functions/variables that were not described in the paper - we have kept them for completeness. Also, we only mention the scripts/functions/variables that were modified by us unless it is crucial in understanding the overall code structure 

### 1. concept_vocabulary
1. cluster_concept.py : used to cluster the concepts 
2. \*concept_classifier.py : used for training a concept classifier (this work was not included in the paper, but we include the code for our release as it was important for our understanding of the concept space prior )
3. \*concept_classifier_dataloader.py : holds dataloader for the concept classifier (this work was not included in the paper)
4. prepare_concept_glove.py : this builds the concept vocabulary from the clustering result from cluster_concept.py
5. \*jupyter_notebook : this folder holds jupyter notebooks used for model analysis, visualization and dataset analysis

### 2. lcgn 
1. \*exp_clevr: this is for CLEVR dataset - this is not used for our work 
2. **exp_gqa** : this folder holds main.py used to run the model with other data files 
3. \*models_clevr: this is for the CLEVR models - this is not used for our work
4. **models_gqa**: this folder holds LCGN models which we have used to implement our three models 
5. util: this folder holds various utility functions - feature loader, etc. -- this was not modified 

#### 2.2 exp_gqa
1. main.py : this file is used to train and test the LCGN model - we also use this file unmodified to train/test our three models

#### 2.4 models_gqa

1. config.py : holds all the hyperparameter settings - here we have added the hyperparaeter settings used for our three models 
   1. CONCEPT_VOCABULARY (bool) : this orders the use of concept vocabulary. For all three models, this is set to True
   1. CPT_TYPE : decides on the model type in ["ORIGINAL", "TENSOR", "RCDI"] for Local-VCS, MCS, Dynamic-VCS respectively. 
   1. ENC_TYPE : type of encoder to use in ["enc-dec", "enc"] - Local-VCS and MCS use "enc" and Dynamic-VCS use "enc-dec"
---
1. config.py continued - for the last two "additional hyperparameters" described in the paper 
   1. CPT_EMB_DIM (int) : embedding size of concept vectors - usually set to 300 
   1. CPT_EMB_INIT_FILE (str): flie location of the concept vocabulary (from 1.4)
   1. INIT_CPT_EMB_FROM_FILE (bool) : whether to randomly initialize the concept vocabulary or not - when True we use the constructed concept vocabulary specified in CPT_EMB_INIT_FILE
   1. NUM_CPT (int) : number of concepts to use - this hold matters when INIT_CPT_EMB_FROM_FILE=False
   1. CPT_EMB_FIXED (bool) : wheter to fix the concept vectors - when True the concept vectors do not accumulate gradients  
   1. CPT_SECTIONS_INIT_FILE (str) : this is auxiliary file used for MCS - this allows the model to know how to seperate the exemplars for different concepts
---
1. config.py continued - for the choice of stem as described in the paper 
   1. NON_LINEAR_STEM (str) : when True - this is used to set the stem to FFN for Local-VCS and MCS
   1. NON_LINEAR_STEM_NUM_LAYERS (int) : number of hidden layers to use for the FFN stem
   1. CPT_NON_LINEAR (bool) : this is used to set the stem to FFN for Dynamic-VCS
---
1. config.py continued - other miscellaneous - used for during development - deprecated mostly  
   1. \*CPT_ATTN_GATE (int) : for Local-VCS, this is used to gate the local-context vector  (to keep the parts that did not match)
   1. \*RETURN_ATTN (bool) : returns attention history - for model analysis
   1. \*ONLY_CPT_OUT (bool) : set to False (this allows the last model to only consider the contextual vectors for classification) 
   1. \*CPT_NON_LINEAR_TYPE : this is always set to RELU or ELU - decides which activate to use for CPT_NON_LINEAR
   1. \*CPT_NON_LINEAR_PROJECTION : this allows for non-linear activation after mapping to the concept space in Dynamic-VCS ( not described in the paper)
   1. \*ADD_LAYER_NORM : adds layernorm at each layer for Dynamic-VCS ( not described in the paper)
   1. \*DIFF_SINGLE_HOP : uses different "hop" operation at the end for aggregation (not descxribed in the paper)
   1. \*INPUT_NUM_LAYERS : number of bi-LSTM encoder layers to use - always set to 1 
   
1. input_units.py : holds encoder related functions for linguistic commands 
   1. EncoderConceptVocabulary : this maps the command to the concept vocabulary thorough attention 
   1. \*EncoderConceptVocabularyTensor : this maps the command to the concept vocabulary with exemplar vectors ( not described in the paper)
   1. EncoderDecoderConceptVocabulary : for Dynamic-VCS
   1. BiLSTMed : BiLSTM torch.nn module that prepares the encoder hidden states for the decoder 

1. lcgn.py : holds all LCGN models (graph message passing algorithm + initialization of local and contextual feature vectors )
   1. LCGNConceptVocabulary : Local-VCS
   1. \*LCGNConceptVocabularyPD : local-VCS model essentially but encoder is only allowed to influence the keys 
   1. \*LCGNConceptVocabularyRC : Dynamic-VCS model but without encoder-decoder controller 
   1. LCGNConceptVocabularyRCDI : Dynamic-VCS
   1. \*LCGNConceptVocabularyRCDIS : Dynamic-VCS model with different message passing algorithm 
   1. LCGNConceptVocabularyTensor : MCS
   
1. model.py : holds the wrapper that combines the encoder + lcgn + aggregation (hop) + classifier 
   1. LCGNnetConceptVocabulary: this was added - this wrapper was introduced for the three models - it uses the same aggregation as the original model

1. ops.py : holds layers and models - was originally written by the LCGN authors to keep the torch version consistent with the tensorflow version
   1. Project : FFN stem with different kind of residual connection 
   1. ClassProject : FFN stem 
   
1. output_unit.py : for classifiers - was not modified 
   
## Train the models 
 
0. Being at directory lcgn, add the root of this repository to ```PYTHONPATH: export PYTHONPATH=.:$PYTHONPATH```

1. Local-VCS with linear stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE Original MSG_ITER_NUM 4 ENC_TYPE enc INIT_CPT_EMB_FROM_FILE True STEM_CNN False CONCEPT_VOCABULARY True CPT_EMB_FIXED True NON_LINEAR_STEM False
 ```
2. Local-VCS with FFN stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE Original MSG_ITER_NUM 4 ENC_TYPE enc INIT_CPT_EMB_FROM_FILE True STEM_CNN False CONCEPT_VOCABULARY True CPT_EMB_FIXED True NON_LINEAR_STEM True
 ```
3. Dynamic-VCS with linear stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE RCDI MSG_ITER_NUM 4 ENC_TYPE enc-dec INIT_CPT_EMB_FROM_FILE True CPT_EMB_FIXED True STEM_CNN False CPT_NON_LINEAR_PROJECTION False
 ```
4. Dynamic-VCS with CNN stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE RCDI MSG_ITER_NUM 4 ENC_TYPE enc-dec INIT_CPT_EMB_FROM_FILE True CPT_EMB_FIXED True STEM_CNN True CPT_NON_LINEAR_PROJECTION False
 ```
5. Dynamic-VCS with FFN stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE RCDI MSG_ITER_NUM 4 ENC_TYPE enc-dec INIT_CPT_EMB_FROM_FILE True CPT_EMB_FIXED True STEM_CNN False CPT_NON_LINEAR_PROJECTION True
 ```
6. MCS with linear stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE TENSOR MSG_ITER_NUM 4 ENC_TYPE enc INIT_CPT_EMB_FROM_FILE True STEM_CNN False CONCEPT_VOCABULARY True CPT_EMB_FIXED True NON_LINEAR_STEM False
 ```
7. MCS with CNN stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE TENSOR MSG_ITER_NUM 4 ENC_TYPE enc INIT_CPT_EMB_FROM_FILE True STEM_CNN True CONCEPT_VOCABULARY True CPT_EMB_FIXED True NON_LINEAR_STEM False
 ```
8. MCS with FFN stem (fixed emb = T, load_emb = T)
 ```
python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True CPT_TYPE TENSOR MSG_ITER_NUM 4 ENC_TYPE enc INIT_CPT_EMB_FROM_FILE True STEM_CNN False CONCEPT_VOCABULARY True CPT_EMB_FIXED True NON_LINEAR_STEM True
 ```


