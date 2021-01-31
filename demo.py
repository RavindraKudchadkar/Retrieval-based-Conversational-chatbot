import sys
from torch import nn
import torch
import numpy as np
from MSN import *
from SMN import *
from DAM import *
from IOI import *
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

with open("demo_utt.txt" ,"r") as f:
	context=f.readline().split("\t")
with open("demo_responses","rb") as f:
	response=pickle.load(f)

fin_response=response

config=ConfigMSN()
model=MSN(config)
model.load_state_dict(torch.load("saved_MSN_model_40_examples.pt"))
model.eval()

with open("word_to_id", "rb") as f:
	word_to_id=pickle.load(f)

def convert_to_ids_demo(word_to_id,context,response):
	context_ids,response_ids=[],[]
	for utt in context:
		cc=[]
		for x in text_to_word_sequence(utt):
			if x.lower() in word_to_id:
				cc.append(word_to_id[x])
			else:
				cc.append(0)  
		context_ids.append(cc)
			   
	for res in response:
		r=[]
		for x in text_to_word_sequence(res):
			if x.lower() in word_to_id:
				r.append(word_to_id[x])
			else:
				r.append(0)  
		response_ids.append(r)
	
	return context_ids,response_ids

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


utterences,responses=convert_to_ids_demo(word_to_id,context,response)

utterences,utt_lens=multi_sequences_padding([utterences])
response_lengths=get_sequences_length(responses, maxlen=50)
responses=pad_sequences(responses,padding='post', maxlen=50)


scores=[]

for (response,res_len) in zip(responses,response_lengths):
	utt,utt_len=utterences[0],utt_lens
	utt=torch.LongTensor(np.array([utt]).astype(np.float32))
	utt_len=torch.LongTensor(np.array(utt_len).astype(np.float32))
	res_len=torch.LongTensor(np.array(res_len).astype(np.float32))
	response=torch.LongTensor(np.array(response).astype(np.float32))
	response=response.unsqueeze(dim=0)
	ypred=model(utt,response,utt_len,res_len)
	score=torch.sigmoid(ypred)
	score=score.squeeze().detach().numpy().tolist()
	scores.append(score)

index=np.argmax(scores,axis=-1)
print(fin_response[index])

