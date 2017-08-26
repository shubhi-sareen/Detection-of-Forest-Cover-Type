from random import randrange
import random
from csv import reader
from random import seed
from math import sqrt
import collections
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

def calcDistance(input_row,row): 
    sum_val = 0.0
    for i in range(len(row)-1):
        sum_val+= (input_row[i]-row[i]) * (input_row[i]-row[i])
    return sqrt(sum_val)

def predict(input_row, train):
    mapy = {}
    for row in train:
        mapy[calcDistance(input_row,row)] = row[-1]
    sorted_by_key = collections.OrderedDict(sorted(mapy.items()))
    count_y = {}
    i = 0
    for k in sorted_by_key:
        if i >= 2:
            break
        if sorted_by_key[k] in count_y:
            count_y[sorted_by_key[k]] += 1
        else:
            count_y[sorted_by_key[k]] = 1
        i+=1
    sorted_by_key_ = collections.OrderedDict(sorted(count_y.items()))
    i = 0
    for k in sorted_by_key_:
        return k

seed(1)
filename = 'dataset.csv'
dataset = load_file(filename)
del dataset[524:]
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
minmax = dataset_minmax(dataset)
normalize(dataset, minmax)
train=[]
test=[]
for i in range(420):
    index = randrange(len(dataset))
    train.append(dataset.pop(index))

test=dataset
count_=0
for row in test:
    pred_value = predict(row,train)
    if pred_value == row[-1]:
        count_+=1
print count_
print len(test)
print len(train)
print float(count_)/len(test)
