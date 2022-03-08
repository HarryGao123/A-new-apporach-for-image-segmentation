#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import confusion_matrix

def evalAccuracy(toolbox, individual, x_train, y_train):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    f1_score = 0.0
    
    for i in range(0, len(y_train)):
        grey = np.array(x_train[i, :, :, :])
       
        output = func(grey)
        output = output.reshape(-1)
        
        true =y_train[i].reshape(-1)

        tn, fp, fn, tp = confusion_matrix(true, output).ravel()
        f1_score += (2 * tp) / ((2 * tp) + fp + fn)
        
    f1_score = round(100*f1_score / len(y_train), 2)
    
    return f1_score,

def evalAccuracy_multiple(toolbox, individual, x_train, y_train):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    f1_score = 0.0
    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    
    for i in range(0, len(y_train)):
        grey = np.array(x_train[i, :, :, :])
       
        output = func(grey)
        output = output.reshape(-1)
        
        true =y_train[i].reshape(-1)
        
        
        tn, fp, fn, tp = confusion_matrix(true, output).ravel()
        total = tp + fp + fn + tn
        
        f1_score += (2 * tp) / ((2 * tp) + fp + fn)
        
        accuracy += (tp + tn) / total
        
        recall += tp/(tp+fn)
        
        precision += tp/(tp+fp)
    
        
    f1_score = round(100*f1_score / len(y_train), 2)
    accuracy = round(100*accuracy / len(y_train), 2)
    recall = round(100*recall / len(y_train), 2)
    precision = round(100*precision / len(y_train), 2)
    
    return f1_score,accuracy, recall, precision,