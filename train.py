from torch import nn
import torch
from torch.nn import init
import numpy as np
import torch.autograd as autograd
import datetime
import pickle
from SMN import *
from DAM import *
from IOI import *
from MSN import *
from load_data import train_loader
import sys
import matplotlib.pyplot as plt 



batch_size=20
lr=0.001
l2_decay=0.0001
epochs=1
model_name=""
args=sys.argv
del args[0]
i=0
while i<len(args):
	if args[i]=="--batch_size":
		i=i+1
		batch_size=int(args[i])
	if args[i]=="--learning_rate":
		i=i+1
		lr=float(args[i])
	if args[i]=="--l2_decay":
		i=i+1
		l2_decay=float(args[i])
	if args[i]=="--epochs":
		i=i+1
		epochs=int(args[i])
	if args[i]=="--model_name":
		i=i+1
		model_name=args[i]
	i=i+1


trainloader=train_loader(batch_size)


def increase_count(correct_count, score, label):
	if ((score.data[0][0] >= 0.5) and (label.data[0][0] == 1.0)) or ((score.data[0][0] < 0.5) and (label.data[0][0]  == 0.0)):
		correct_count +=1  
	return correct_count

def get_accuracy(correct_count, dataframe_length,batch_size):
	accuracy = correct_count/(dataframe_length)*batch_size
	return accuracy

def train_model(trainloader,lr,l2_penalty,epochs,model_name):
	config,model=None,None
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


	optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08,weight_decay = l2_penalty)
	loss_func = torch.nn.BCEWithLogitsLoss()
	accuracy_train=[]
	loss_train=[]
	epochss=[]
	model.train()
	for epoch in range(epochs): 
		sum_loss_training = 0.0
		training_correct_count = 0
		for i,(context,response,label,context_lens,response_lens) in enumerate(trainloader):
			model.zero_grad()
			context = autograd.Variable(context, requires_grad = False) 
			response = autograd.Variable(response, requires_grad = False) 
			label = autograd.Variable(label.unsqueeze(1), requires_grad = False) 
			score=model(context,response,context_lens,response_lens)
			score=autograd.Variable(score,requires_grad=True)
			loss = loss_func(score, label)
			sum_loss_training += loss.item()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			training_correct_count = increase_count(training_correct_count, score, label)
		loss_train.append(sum_loss_training)                                        
		training_accuracy = get_accuracy(training_correct_count, len(trainloader)*batch_size,batch_size)
		epochss.append(epoch)
		accuracy_train.append(training_accuracy)
		print(str(datetime.datetime.now()).split('.')[0], "Epoch: %d/%d" %(epoch,epochs),"TrainLoss: %.3f" %(sum_loss_training/(len(trainloader)*batch_size)),"TrainAccuracy: %.3f" %(training_accuracy))  
	torch.save(model.state_dict(), 'saved_'+model_name+'_model_%d_examples.pt' %(len(trainloader)*batch_size))
	return epochss,accuracy_train,loss_train


epochs,accuracy_train,loss_train=train_model(trainloader,lr,l2_decay,epochs,model_name)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs,accuracy_train,'y',label='training accuracy')
plt.title('Training loss and accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/accuracy')
plt.legend()
plt.savefig(model_name+"_plot.png")