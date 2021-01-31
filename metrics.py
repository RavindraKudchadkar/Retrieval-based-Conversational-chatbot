import numpy as np
import pickle

def recall_at_k(labels, scores, k=5, num_of_utt=10):
  scores = scores.reshape(-1, num_of_utt)
  labels = labels.reshape(-1, num_of_utt) 
  sorted, indices = np.sort(scores, 1), np.argsort(-scores, 1)
  count_nonzero = 0
  recall = 0
  for i in range(indices.shape[0]):
    num_rel = np.sum(labels[i])
    if num_rel==0: continue
    rel = 0
    for j in range(k):
      if labels[i, indices[i, j]] == 1:
        rel += 1
    recall += float(rel) / float(num_rel)
    count_nonzero += 1
  return float(recall) / count_nonzero


def precision_at_k(labels, scores, k=1, num_of_utt=10):
  scores = scores.reshape(-1,num_of_utt) 
  labels = labels.reshape(-1,num_of_utt) 
  sorted, indices = np.sort(scores, 1), np.argsort(-scores, 1)
  count_nonzero = 0
  precision = 0
  for i in range(indices.shape[0]):
    num_rel = np.sum(labels[i])
    if num_rel==0: continue
    rel = 0
    for j in range(k):
      if labels[i, indices[i, j]] == 1:
        rel += 1
    precision += float(rel) / float(k)
    count_nonzero += 1
  return precision / count_nonzero


def MRR(target, logits, k=10):
  assert logits.shape == target.shape
  target = target.reshape(-1,k)
  logits = logits.reshape(-1,k)
  sorted, indices = np.sort(logits, 1)[::-1], np.argsort(-logits, 1)
  count_nonzero=0
  reciprocal_rank = 0
  for i in range(indices.shape[0]):
    flag=0
    for j in range(indices.shape[1]):
      if target[i, indices[i, j]] == 1:
        reciprocal_rank += float(1.0) / (j + 1)
        flag=1
        break
    if flag: count_nonzero += 1
  return float(reciprocal_rank) / count_nonzero



def MAP(target, logits, k=10):
    
    assert logits.shape == target.shape

    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)
    
    sorted, indices = np.sort(logits, 1)[::-1], np.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += float(num_rel) / (j + 1)
        if num_rel==0: continue
        average_precision = average_precision / num_rel
    
        map_sum += average_precision
        count_nonzero += 1
 
    return float(map_sum) / count_nonzero
