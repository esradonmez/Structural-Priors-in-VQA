# import re
#
# _SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
#
#
# def tokenize(sentence):
#     tokens = _SENTENCE_SPLIT_REGEX.split(sentence.lower())
#     tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
#     return tokens
#import gensim.downloader as api

def tokenize_gqa(sentence,
                 ignoredPunct=["?", "!", "\\", "/", ")", "("],
                 keptPunct=[".", ",", ";", ":"]):
    sentence = sentence.lower()
    for punct in keptPunct:
        sentence = sentence.replace(punct, " " + punct + " ")
    for punct in ignoredPunct:
        sentence = sentence.replace(punct, "")
    tokens = sentence.split()
    return tokens


tokenize_clevr = tokenize_gqa


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

#modified------------------------------------------------------------------
def load_json_dict(fname):
    dic = json.load(open(fname, 'r'))
    return dic

#modified------------------------------------------------------------------

class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (
            self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict
            else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does'
                             ' not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds

#modified------------------------------------------------------------------
class ConceptDict:

    def __init__(self, concept_vocab_file, glove_name):

        # #initialize word_list -- not included for now 
        # super(ConceptDict, self).__init__(self, word_list)

        print('loading concept vocabulary from %s' % concept_vocab_file)
        self.attr2cncpt = load_json_dict(concept_vocab_file) 
        self.cncpt2attr = {v:k  for k, v in self.attr2cncpt}

        #initialize concept embeddings here
        print('loading glove %s' % glove_name)
        glove = api.load(glove_name)
        H = glove.vector_size
        self.cncpt_emb = {cncpt:{} for cncpt in self.cncpt2attr.keys()}
        for cncpt, attr in self.cncpt2attr.items():
            attr_tokens = attr.strip().split(' ')
            #for the attributes with more than one token we average them 
            concept_emb = np.zeros((len(attr_tokens), H), dtype=np.float32)
            zero_vector = np.zeros(H, dtype=np.float32)
            for i, oken in enumerate(attr_tokens):
                concept_emb[i, :]= glove.get(attr, zero_vector)
            self.cncpt_emb[cncpt][attr] = concept_emb
    
        self.num_vocab = len(self.cncpt_emb)
        self.cncpt_emb_size = H

#modified------------------------------------------------------------------