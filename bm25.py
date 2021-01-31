import sys
import numpy as np
import math
from collections import Counter
import pickle


def weight(count, dfs, normal, term):
	if term not in dfs:
		return 0
	idf = math.log((100 + 0.1 - dfs[term])/(dfs[term] + 0.1) + 1)
	tf = count[term]
	k = 1.5
	b = 0.251
	w = idf * tf * (k + 1) / (tf + k * (1 + b - b * normal/docavg))
	return w


def preprocessing(texts):
    new_text = remove_punct(texts)
    new_text = stemmer(new_text)
    return new_text


def remove_punct(input):
    symbols = "!\"#$%&()*+-,.:;=?@[\]^_<?>'`{|}~\n"
    for i in symbols:
        input = np.char.replace(input, i, ' ')
    return input


def stemmer(input):
    text = str(input)
    ret = ""
    # ks = krovetz.PyKrovetzStemmer()
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    for word in text.split():
        ret += " "
        # word = ks.stem(word)
        if word in stop:
            continue
        ret += word
    return ret

docs = []
queries = []

with open("test1000.txt", "r", encoding="utf8") as f:
	line = f.readline()
	line1 = line.split('\t')
	query = ""
	for i in range(1, len(line1)-1):
		query += line1[i]
		query += " "
	queries.append(query)
	docs.append(line1[len(line1)-1])
	for i in range(0, 9):
		line = f.readline()
		line1 = line.split('\t')
		doc = line1[len(line1)-1]
		docs.append(doc)
	while line:
		line = f.readline()
		line1 = line.split('\t')
		query = ""
		for i in range(1, len(line1)-1):
			query += line1[i]
			query += " "
		# print(query)
		queries.append(query)
		docs.append(line1[len(line1)-1])
		for i in range(0, 9):
			line = f.readline()
			line1 = line.split('\t')
			doc = line1[len(line1)-1]
			docs.append(doc)
f.close()
# print(len(queries))
# print(len(docs))
scores = {}
for i in range(0, len(queries)):
	print(i+1)
	table = []
	df = {}
	dl = {}
	docavg = 0
	qid = queries[i]
	for j in range(10*i, 10*i+10):
		text = docs[j]
		text = preprocessing(text)
		text = text.split()
		count = Counter(text)
		table.append([j, count])
		docLength = 0
		for word in count:
			if word not in df:
				df[word] = 1
			else:
				df[word] = df[word] + 1
			docLength += count[word]
		dl[j] = docLength
		docavg += docLength
	docavg /= 10
	qid = preprocessing(qid)
	qLen = len(qid.split())
	for document in table:
		scores[document[0]] = 0
		for term in qid.split():
			scores[document[0]] += weight(document[1], df, dl[document[0]], term)

finalscore = []
for ele in scores:
	finalscore.append(scores[ele])

finalscore = np.array(finalscore)
finalscore = np.exp(-finalscore)
finalscore = 1 / (1 + finalscore)
finalscore = finalscore.reshape(-1,10)
shape = finalscore.shape
finalscore = finalscore.tolist()
# print(finalscore)
fila = open("bm25_output", "wb")
pickle.dump(finalscore, fila)
fila.close()

lables = np.zeros((shape[0], shape[1]))
lables[:,0] = 1
lables = lables.tolist()
fila = open("test_label", "wb")
pickle.dump(lables, fila)
fila.close()

# print(lables)