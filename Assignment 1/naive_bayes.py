# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:03:58 2019

@author: Skanda
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB 


def parse(filename, word_list, sentences, labels):
    words = []
    with open(filename, 'r', encoding = 'utf-8') as fp:
        for line in fp:
            splitted = line.split('.txt ')
            if splitted[0].split(' ')[1] == 'pos':
                labels.append(1)
            else:
                labels.append(0)
            splitted[1] = splitted[1].replace('\n', '')
            sentences.append(splitted[1])
            words = splitted[1].split()
            word_list += words
      
        

def fill_data(sentences):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    i = 0
    for d in sentences:
        for term in d.split():
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
        i = i + 1
        print(i)

    sparse_data = sparse.csr_matrix((data, indices, indptr), dtype=int)
    return sparse_data
    
     
def main():
    '''
    vocabulary = []
    word_list = []
    sentences = []
    labels = []
    parse('naive_bayes_data.txt', word_list, sentences, labels)
    vocabulary = list(set(word_list))
    with open('vocabulary.pickle', 'wb') as fp:
        pickle.dump(vocabulary, fp)
    with open('sentences.pickle', 'wb') as fp:
        pickle.dump(sentences, fp)
    with open('labels.pickle', 'wb') as fp:
        pickle.dump(labels, fp)
    vocabulary = []
    sentences = []
    labels = []
    with open('vocabulary.pickle', 'rb') as fp:
        vocabulary = pickle.load(fp)
    with open('sentences.pickle', 'rb') as fp:
        sentences = pickle.load(fp)
    with open('labels.pickle', 'rb') as fp:
        labels = pickle.load(fp)
    data = fill_data(sentences)
    sparse.save_npz('data.npz', data)
    del vocabulary
    del sentences
    '''
    labels = []
    with open('labels.pickle', 'rb') as fp:
        labels = pickle.load(fp)
    data = sparse.load_npz('data.npz')
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    class1 = X_train[y_train == 1]
    class0 = X_train[y_train == 0]
    prob_c1 = class1.shape[0]/X_train.shape[0]
    prob_c0 = 1 - prob_c1
    prob_values_class1 = []
    prob_values_class0 = []
    class1_transpose = class1.T
    class0_transpose = class0.T
    words_c1 = 0
    words_c0 = 0
    for row in class1_transpose:
        if row.sum() > 0:
            words_c1 += row.sum()
    for row in class0_transpose:
        if row.sum() > 0:
            words_c0 += row.sum()
    for row in class1_transpose:
        prob_values_class1.append((row.sum() + 1)/(words_c1 + class1.shape[1]))
    for row in class0_transpose:
        prob_values_class0.append((row.sum() + 1)/((words_c0 + class0.shape[1])))
    y_preds = []
    X_test = X_test.toarray()
    count = 1
    for row in X_test:
        y1 = prob_c1
        y0 = prob_c0
        i = 0
        for value in row:
            if value > 0:
                y1 *= prob_values_class1[i]**value
                y0 *= prob_values_class0[i]**value
            i = i + 1
        if y1 >= y0:
            y_preds.append(1)
        else:
            y_preds.append(0)
        print(count)
        count = count + 1
    y_preds = np.array(y_preds)
    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    accuracy = (tn + tp)/(tn + tp + fn + fp)
    print(accuracy)
    print(confusion_matrix(y_test, y_preds))
    
                
        
        
    
	
    
    
if __name__ == '__main__':
    main()
