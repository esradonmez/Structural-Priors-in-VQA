
from concept_classifier_dataloader import prepare_concept_vocabulary
from pymagnitude import Magnitude
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize

dim = 300
globe_vectors = Magnitude("glove.6B.{}d.magnitude".format(dim))
nltk.download('punkt')

def pool_glove(words, pooling = 'sum'):
    '''
    creates an embedding for the given key-relation-otherkey 
    '''

    vecs = [globe_vectors.query(word).reshape(1, -1) for word in words]
    vecs = np.stack(vecs, axis=0)
    
    if pooling == 'sum':
        vec = np.sum(vecs, axis=0)
    elif pooling == 'product':
        vec = np.prod(vecs, axis=0)
    elif pooling == 'max':
        vec = np.max(vecs, axis=0)
    else:
        vec = np.mean(vecs, axis=0)
    return vec

def collect_gloves(extracted):
    '''
    this is the function that collects all concepts 
    '''
    all_vectors = []

    for w, c, r, s in extracted:
        words = w.strip().split(' ')
        vec = pool_glove(words, pooling='sum')
        all_vectors.append(vec)
        cptMap.append((w, c, r, s))

    all_vectors = np.stack(all_vectors, axis=0).unsqueeze()
    return all_vectors, cptMap

# this prepares the concept vocabulary - as a matrix 
# concept_vocab, val2concept, concept2val = prepare_concept_vocabulary(path='data/concepts.json')


concept2val = json.load(open(
    '/mount/studenten/arbeitsdaten-studenten1/advanced_ml/sgg_vqa_je/grouped_concepts_300.json', 'r'))
concept2val = {int(k)+1:v for k, v in concept2val.items()}
concept2val[0] = [name.strip()
                                 for name in open('AML-group-3/lcgn/exp_gqa/data/name_gqa.txt', 'r').readlines()]
val2concept = {value:k for k, v in concept2val.items() for value in v }

concept_glove_vectors = np.zeros((len(concept2val), dim))
if True:
    print('Number of concepts are {}'.format(len(concept2val)))
    for cptIdx, cptVals in concept2val.items():
        print(cptIdx)
        if int(cptIdx) == 0:
            continue
        cptVals = [val for val in cptVals if val != 'none']
        cptVector = np.stack([pool_glove(val) for val in cptVals if val != 'none']).mean()
        concept_glove_vectors[int(cptIdx), :] = cptVector

    with open('concept.{}d.npy'.format(dim), 'wb') as f:
        np.save(f, concept_glove_vectors)

# this prepares the concept vocbulary in terms of tensors 
print(concept2val)
#print(max((val for val in concept2val.values())))
keys = sorted(list(concept2val.keys()))
print(keys)
#all_concepts_size =sum(len(concept2val[key])-1  if key != 0 else concept2val[key] for key in keys )
all_concepts_size =sum(len(concept2val[key]) for key in keys )
print(all_concepts_size)

# first holds the concept vectors
W = np.zeros((len(concept2val) + all_concepts_size, dim))
W[:len(concept2val)] = concept_glove_vectors

#object names 
object_names = [ name.strip() for name in open('AML-group-3/lcgn/exp_gqa/data/name_gqa.txt', 'r').readlines()]

i = len(concept2val)
concept_sections = [i]
for key in keys:
    i = 0
    if key == 0:
        for name in object_names:
            words = name.strip().split(' ')
            vec = pool_glove(words, pooling='sum') 
            W[i, :] = vec
            i+=1


    else:
        vals = concept2val[key]
        for val in vals:
            if val != 'none':
                words = val.strip().split(' ')
                vec = pool_glove(words, pooling='sum')
                W[i, :] = vec
                i += 1 
    
    concept_sections.append(i)

with open('concept_tensor_human.{}d.npy'.format(dim), 'wb') as f:
    np.save(f, W)

with open('concept_sections_human.{}d.npy'.format(dim), 'wb') as f:
    np.save(f, concept_sections)

print(W.shape)
print(sum(concept_sections))
