from random import randrange
from random import random
from csv import reader
from random import seed
from math import exp

def load_file(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
	row[column] = lookup[row[column]]
    return lookup

def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column]:
            row[column] = float(row[column].strip())

def dataset_minmax(dataset):
    minmax = list()
    statistics = [[min(column), max(column)] for column in zip(*dataset)]
    return statistics

def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i]-minmax[i][0])/(minmax[i][1] - minmax[i][0])   
 
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
	while len(fold) < fold_size:
	    index = randrange(len(dataset_copy))
       	    fold.append(dataset_copy.pop(index))
	dataset_split.append(fold)
    return dataset_split
 
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
	    correct += 1
    return correct / float(len(actual)) * 100.0
 

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
	train_set.remove(fold)
	train_set = sum(train_set, [])
	test_set = list()
	for row in fold:
	    row_copy = list(row)
	    test_set.append(row_copy)
	    row_copy[-1] = None
	predicted = algorithm(train_set, test_set, *args)
	actual = [row[-1] for row in fold]
	accuracy = accuracy_metric(actual, predicted)
	scores.append(accuracy)
    return scores 
       
def initialize(input_layer_size, hidden_layer_size, output_layer_size):
    nn = list()
    hidden_layer = [{'weights':[random() for i in range(input_layer_size + 1)]} for i in range(hidden_layer_size)]
    nn.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(hidden_layer_size + 1)]} for i in range(output_layer_size)]
    nn.append(output_layer)
    return nn

def activate(weights, input_layer):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * input_layer[i]
    return activation

def sigmoid(activation):
    return 1.0/(1.0 + exp(-activation))

def forward_propagation(nn, row):
    inputs = row
    for layer in nn:
        new_input = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_input.append(neuron['output'])
        inputs = new_input
    return inputs

def derivative(output):
    return output*(1.0-output)

def backpropogate_error(nn,expected):
    for i in reversed(range(len(nn))):
        layer = nn[i]
        errors = list()
        if i != len(nn) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in nn[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derivative(neuron['output'])

def weight_update(nn, row, learning_rate):
    for i in range(len(nn)):
        inputs = row[:-1]
	if i != 0:
	    inputs = [neuron['output'] for neuron in nn[i - 1]]
        for neuron in nn[i]:
	    for j in range(len(inputs)):
		neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
	    neuron['weights'][-1] += learning_rate * neuron['delta'] 

def trainNetwork(nn, train, learning_rate, epochs, output_size):
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            output = forward_propagation(nn, row)
            expected = [0 for i in range(output_size)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
	    backpropogate_error(nn, expected)
	    weight_update(nn, row, learning_rate)


def predict(nn, row):
    outputs = forward_propagation(nn, row)
    return outputs.index(max(outputs))

def backpropagation(train, test, learning_rate, epochs, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize(n_inputs, n_hidden, n_outputs)
    trainNetwork(network, train, l_rate, epochs, n_outputs)
    predictions = list()
    for row in test:
	prediction = predict(network, row)
	predictions.append(prediction)
    return(predictions)
seed(1)
filename = 'dataset.csv'
dataset = load_file(filename)
del dataset[524:]
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
minmax = dataset_minmax(dataset)
normalize(dataset, minmax)

n_folds = 6
l_rate = 0.3
n_epoch = 500
n_hidden = 20
scores = evaluate_algorithm(dataset, backpropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
