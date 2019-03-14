#!/usr/bin/python3

# use natural language toolkit
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import json
import time
import datetime

stemmer = SnowballStemmer("italian")

# training data
training_file = "training.json"
with open(training_file) as data_file:
	training_data = json.load(data_file)
	
# training_data = []
# training_data.append({"class":"Sai tutto", "sentence":"Come si fa?"})
# training_data.append({"class":"Sai tutto", "sentence":"Non ho studiato"})
# training_data.append({"class":"Sai tutto", "sentence":"Esame"})

# training_data.append({"class":"Stai zitto", "sentence":"Non so niente"});

# training_data.append({"class":"Ma What?", "sentence":"Ruote dentate"})
# training_data.append({"class":"Ma What?", "sentence":"Panino al pollo"})
# training_data.append({"class":"Ma What?", "sentence":"Avocado"})
# training_data.append({"class":"Ma What?", "sentence":"C'é un bot di Giorgio"})

# training_data.append({"class":"Che nerd", "sentence":"Devo andare a girare in bici"})
# training_data.append({"class":"Che nerd", "sentence":"Anatra"});

# training_data.append({"class":"Vita perfetta", "sentence":"Sono con la mia ragazza"})

words = []
classes = []
documents = []
ignore_words = ['?', '\'']
# loop through each sentence in the training data
for pattern in training_data:
	# tokenize each work in the sentence
	w = nltk.word_tokenize(pattern['sentence'])
	# add to words list
	words.extend(w)
	# add to documents in corpus
	documents.append((w, pattern['class']))
	# add to classes list
	if pattern['class'] not in classes:
		classes.append(pattern['class'])

# stem and lower each work and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "words", words)

# create training data
training = []
output = []
# create an empty array for output data
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
	# initialize our bag of words
	bag = []
	# list of tokenized words for the pattern
	pattern_words = doc[0]
	# stem each word
	pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
	# create a bag of words array
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	training.append(bag)
	# output is a '0' for each tag and '1' for current tag
	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1
	output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]

# compute sigmoid nonlinearity
def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
	return output*(1-output)

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

	print("Training with %s neurons, alpha:%s, droput:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
	print("Input matrix: %s%s - Output matrix: %s%s" % (len(X),len(X[0]),1,len(classes)))
	np.random.seed(1)

	last_mean_error = 1
	# randomly initialize out weights with mean 0
	synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
	synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

	prev_synapse_0_weight_update = np.zeros_like(synapse_0)
	prev_synapse_1_weight_update = np.zeros_like(synapse_1)

	synapse_0_direction_count = np.zeros_like(synapse_0)
	synapse_1_direction_count = np.zeros_like(synapse_1)

	for j in iter(range(epochs+1)):

		# feed forward through layers 0, 1 and 2
		layer_0 = X
		layer_1 = sigmoid(np.dot(layer_0, synapse_0))

		if (dropout):
			layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1-dropout_percent)[0] * (1.0/(1-dropout_percent))

		layer_2 = sigmoid(np.dot(layer_1, synapse_1))

		# how much did we miss the target value
		layer_2_error = y - layer_2

		if (j%10000) == 0 and j > 5000:
			# if this 10k iteration's error is greater than the last iteration, break out
			if np.mean(np.abs(layer_2_error)) < last_mean_error:
				print("delta after "+str(j)+" iterations:"+str(np.mean(np.abs(layer_2_error))))
				last_mean_error = np.mean(np.abs(layer_2_error))
			else:
				print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
				break

		layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
		
		layer_1_error = layer_2_delta.dot(synapse_1.T)
		layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

		synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
		synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

		if (j > 0):
			synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0)+0))
			synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0)+0))

		synapse_1 += alpha * synapse_1_weight_update
		synapse_0 += alpha * synapse_0_weight_update

		prev_synapse_0_weight_update = synapse_0_weight_update
		prev_synapse_1_weight_update = synapse_1_weight_update

	now = datetime.datetime.now()

	# persist synapses
	synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
			   'datetime': now.strftime("%Y-%m-%d %H:%M"),
			   'words': words,
			   'classes': classes
			   }
	synapse_file = "synapses.json"

	with open(synapse_file, 'w') as outfile:
		json.dump(synapse, outfile, indent=4, sort_keys=True)

	print("Saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=40, alpha=0.1, epochs=200000, dropout=False, dropout_percent=0.1)

elapsed_time = time.time() - start_time
print("Processing time:", elapsed_time, "seconds")