import torch
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import pickle
import sys
from collections import Counter

train_file,test_file,glovefile="","",""
args=sys.argv
del args[0]
i=0
while i<len(args):
  if args[i]=="--trainpath":
    i=i+1
    train_file=args[i]
  if args[i]=="--testpath":
    i=i+1
    test_file=args[i]
  if args[i]=="--glovepath":
    i=i+1
    glovefile=args[i]
  i=i+1


def create_stuff_traindata(file_name):
    file=open(file_name,"r",encoding="utf-8")
    line=file.readline()
    labels,context,response,single_turn_context=[],[],[],[]
    vocab = []
    word_freq = {}
    while line:
        arr=line.strip().split(sep="\t")
        label=arr[0]
        utterances=arr[1:-2]
        answer=arr[-1]
        labels.append(label)
        context.append(utterances)
      
        single_turn_context.append(str(" ".join([str(ele) for ele in utterances])))
        prf=Counter(str(" ".join([str(ele) for ele in utterances])).split())
        keys=prf.most_common(10)
        stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        for (key,val) in keys:
            if key.lower() not in stopwords:
                answer=answer+" "+ key

        response.append(answer)
        train_words = text_to_word_sequence(str(" ".join([str(ele) for ele in utterances])) + " "+ str(answer))
        for word in train_words:
            if word.lower() not in vocab:
                vocab.append(word.lower())         
            if word.lower() not in word_freq:
                word_freq[word.lower()] = 1
            else:
                word_freq[word] += 1
        line=file.readline()
        
    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = ["<UNK>"] + [pair[0] for pair in word_freq_sorted]
    
    word_to_id = {word: id for id, word in enumerate(vocab)}
    return context,response,labels,vocab,word_to_id,single_turn_context

def create_stuff_testdata(file_name):
  file=open(file_name,"r",encoding="utf-8")
  line=file.readline()
  labels,context,response,single_turn_context=[],[],[],[]
  vocab = []
  word_freq = {}
  while line:
      arr=line.strip().split(sep="\t")
      label=arr[0]
      utterances=arr[1:-2]
      answer=arr[-1]
      labels.append(label)
      context.append(utterances)
      response.append(answer)
      single_turn_context.append(str(" ".join([str(ele) for ele in utterances])))
      line=file.readline()
  return context,response,labels ,single_turn_context

def convert_to_ids(word_to_id,context,response,single_turn_context):
    context_ids,response_ids,single_turn_ids=[],[],[]
    for arr in context:
        c=[]
        for utt in arr:
          cc=[]
          for x in text_to_word_sequence(utt):
            if x.lower() in word_to_id:
              cc.append(word_to_id[x])
            else:
              cc.append(0)  
          c.append(cc)
        context_ids.append(c)    
    
    
    for res in response:
      r=[]
      for x in text_to_word_sequence(res):
        if x.lower() in word_to_id:
          r.append(word_to_id[x])
        else:
          r.append(0)  
      response_ids.append(r)

    for res in single_turn_context:
      r=[]
      for x in text_to_word_sequence(res):
        if x.lower() in word_to_id:
          r.append(word_to_id[x])
        else:
          r.append(0)  
      single_turn_ids.append(r)
    
    return context_ids,response_ids,single_turn_ids

def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length

def multi_sequences_padding(all_sequences, max_sentence_len=50):
    max_num_utterance = 10
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, maxlen=max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length
    
def create_id_to_vec(word_to_id, glovefile): 
    lines = open(glovefile, 'r',encoding="utf-8").readlines()
    id_to_vec = {}
    vector = None
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32')
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
    
    for word, id in word_to_id.items(): 
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
    
    embedding_weights = torch.FloatTensor(len(id_to_vec), len(id_to_vec[1]) )
    for id, vec in id_to_vec.items():
        embedding_weights[id] = vec
    embedding_dim=id_to_vec[0].shape[0]
    with open("word_embeddings","wb") as f:
      pickle.dump(embedding_weights.detach().numpy().tolist(),f)
    return embedding_dim


def create_train_files(train_file,glovefile):
    context,response,label,vocab,word_to_id,single_turn_context=create_stuff_traindata(train_file)
    print(len(vocab))
    emb_dim=create_id_to_vec(word_to_id,glovefile)
    context,response,single_turn_context=convert_to_ids(word_to_id,context,response,single_turn_context)
    context,context_lengths=multi_sequences_padding(context)
    with open("context","wb") as f:
      pickle.dump(context,f)

    with open("context_lengths","wb") as f:
      pickle.dump(context_lengths,f) 

    response_lengths=get_sequences_length(response, maxlen=50)
    response=pad_sequences(response,padding='post', maxlen=50)
    single_turn_context=pad_sequences(single_turn_context,padding="post",maxlen=150)

    with open("response", "wb") as f:
      pickle.dump(response,f)

    with open("response_lengths","wb") as f:
      pickle.dump(response_lengths,f)

    with open("label","wb") as f:
      pickle.dump(label,f)

    with open("vocab","wb") as f:
      pickle.dump(vocab,f)

    with open("single_turn_context","wb") as f:
      pickle.dump(single_turn_context,f)

    with open("word_to_id","wb") as f:
      pickle.dump(word_to_id,f)

    return word_to_id

def create_test_files(test_file,word_to_id):
    context,response,label,single_turn_context=create_stuff_testdata(test_file)
    context,response,single_turn_context=convert_to_ids(word_to_id,context,response,single_turn_context)
    context,context_lengths=multi_sequences_padding(context)
    response_lengths=get_sequences_length(response, maxlen=50)
    response=pad_sequences(response,padding='post', maxlen=50)
    single_turn_context=pad_sequences(single_turn_context,padding="post",maxlen=150)

    with open("test_context","wb") as f:
        pickle.dump(context,f)

    with open("test_context_lengths","wb") as f:
      pickle.dump(context_lengths,f)

    with open("test_response", "wb") as f:
        pickle.dump(response,f)

    with open("test_response_lengths","wb") as f:
      pickle.dump(response_lengths,f)

    with open("test_label","wb") as f:
      pickle.dump(label,f)

    with open("test_single_turn_context","wb") as f:
      pickle.dump(single_turn_context,f)

    return

word_to_id=create_train_files(train_file,glovefile)
create_test_files(test_file,word_to_id)