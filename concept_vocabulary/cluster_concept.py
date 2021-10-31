from sklearn.cluster import k_means
import json
import numpy as np

class ConceptSpace:
  def __init__(self, glove_dir, embedding_dim, scene_graph_train, scene_graph_val):
    self.embedding_file = glove_dir + "/glove.6B." + str(embedding_dim) +"d.txt"
    with open(scene_graph_train) as f:
      self.train = json.load(f)
    with open(scene_graph_val) as f:
      self.val = json.load(f)
    self.embedding_dim = embedding_dim

    object_vocab = set()
    for k, _v in self.train.items():
      for o, v in _v["objects"].items():
        object_vocab.add(v["name"])

    for k, _v in self.val.items():
      for o, v in _v["objects"].items():
        object_vocab.add(v["name"])

    attribute_vocab = set()
    for k, _v in self.train.items():
      for o, v in _v["objects"].items():
        for a in v["attributes"]:
          attribute_vocab.add(a)
    
    for k, _v in self.val.items():
      for o, v in _v["objects"].items():
        for a in v["attributes"]:
          attribute_vocab.add(a)
    
    relation_vocab = set()
    for k, _v in self.train.items():
      for o, v in _v["objects"].items():
        for r in v["relations"]:
          relation_vocab.add(r["name"])
    
    for k, _v in self.val.items():
      for o, v in _v["objects"].items():
        for r in v["relations"]:
          relation_vocab.add(r["name"])
  
    print("Scene graphs contain {} objects, {} attributes and {} relations.".format(len(object_vocab), 
                                                                                    len(attribute_vocab), 
                                                                                    len(relation_vocab)))
    
    self.objects = list(object_vocab)
    self.attributes = list(attribute_vocab)
    self.relations = list(relation_vocab)
  
  def create_embeddings(self):
    embeddings_dict = {}
    with open(self.embedding_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    
    embeddings = []
    for l in self.attributes:
      words = l.split(" ")
      if len(words) > 1:
        summed = np.zeros(self.embedding_dim)
        for w in words:
          if w in embeddings_dict.keys():
            embedding = embeddings_dict[w]
            summed += embedding
        embeddings.append(summed/len(words))
      else:
        embeddings.append(embeddings_dict[words[0]])

    return embeddings   

  def cluster(self, n_clusters):
    embeddings = self.create_embeddings()
    
    return k_means(X=embeddings, n_clusters=n_clusters, init='k-means++', max_iter=300)