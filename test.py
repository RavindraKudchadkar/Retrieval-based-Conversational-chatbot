from torch import nn
import torch
import numpy as np
import torch.autograd as autograd
import pickle
from SMN import *
from DAM import *
from IOI import *
from MSN import *
import biLSTM
from load_data import test_loader,single_turn_test_loader
import sys
from torch.utils.data import TensorDataset,DataLoader

model_name=""
model_path=""
args=sys.argv
del args[0]
i=0
while i<len(args):
	if args[i]=="--model_name":
		i=i+1
		model_name=args[i]
	if args[i]=="--model_path":
		i=i+1
		model_path=args[i]
	i=i+1

model=None
if model_name=="SMN":
	config=ConfigSMN()
	model=SMN(config)

if model_name=="DAM":
	config=ConfigDAM()
	model=DAM(config)

if model_name=="IOI":
	config=ConfigIOI()
	model=IOI(config)

if model_name=="MSN":
	config=ConfigMSN()
	model=MSN(config)

if "LSTM" in model_name:
	config=biLSTM.ConfigLSTM()
	encoder=biLSTM.Encoder(config)
	model=biLSTM.DualEncoder(encoder)


model.load_state_dict(torch.load(model_path))
model.eval()



def prediction(context,response,context_lens,response_lens):
  context,response=autograd.Variable(context),autograd.Variable(response)
  test=TensorDataset(context,response,context_lens,response_lens)
  test_loader=DataLoader(test,batch_size=50)
  scores=[]
  for i,(context,response,context_lens,response_lens) in enumerate(test_loader):
    ypred=model(context,response,context_lens,response_lens)
    score=torch.sigmoid(ypred).detach().numpy().tolist()
    scores.append(score)
  return np.array(scores).reshape(-1,10)

def prediction_single_turn(context,response):
  context,response=autograd.Variable(context),autograd.Variable(response)
  test=TensorDataset(context,response)
  test_loader=DataLoader(test,batch_size=50)
  scores=[]
  for i,(context,response) in enumerate(test_loader):
    ypred=model(context,response)
    score=torch.sigmoid(ypred).detach().numpy().tolist()
    scores.append(score)
  return np.array(scores).reshape(-1,10)

if model_name=="SMN" or model_name=="DAM" or model_name=="IOI" or model_name=="MSN" :
  context,response,label,context_lens,response_lens=test_loader()
  scores=prediction(context,response,context_lens,response_lens)

elif "LSTM" in model_name:
  context,response,label=single_turn_test_loader()
  scores=prediction_single_turn(context,response)

with open("scores_of_"+model_name,"wb") as f:
	pickle.dump(scores,f)
