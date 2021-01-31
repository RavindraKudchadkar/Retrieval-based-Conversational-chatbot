from metrics import *
import pickle
import sys

args=sys.argv
del args[0]
score_file=args[1]

with open("test_label", "rb") as f:
	label=np.array(pickle.load(f)).reshape(-1,10).astype(np.float32)

with open(score_file,"rb") as f:
	scores=np.array(pickle.load(f)).astype(np.float32)

recall_at_1=recall_at_k(label,scores,k=1)
recall_at_2=recall_at_k(label,scores,k=2)
recall_at_5=recall_at_k(label,scores,k=5)

precision_at_1=precision_at_k(label,scores,k=1)
precision_at_2=precision_at_k(label,scores,k=2)
precision_at_5=precision_at_k(label,scores,k=5)

mrr_score=MRR(label,scores)
map_score=MAP(label,scores)

print("R@1 : ",recall_at_1)
print("R@2 : ",recall_at_2)
print("R@5 : ",recall_at_5)

print("P@1 : ",precision_at_1)
print("P@2 : ",precision_at_2)
print("P@5 : ",precision_at_5)

print("MRR : ", mrr_score)
print("MAP : ", map_score)



