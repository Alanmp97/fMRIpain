# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:17:47 2023

@author: damia
"""

import warnings
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

def confusion_matrix(y_test, pred):
    #Construct the Confusion Matrix
    labels  = ['NAIVE', 'CPH']
    cm = confusion_matrix(y_test, pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()
    
    
def ROC(self,model,X_test,y_test, l):
    #Classification Area under curve
     warnings.filterwarnings('ignore')
     
     # predict probabilities
     probs = model.predict(X_test, steps = 620*l)
     # keep probabilities for the positive outcome only
     probs = probs[:, 1]
     
     for i in range(len(probs)):
         if probs[i] >= 0.5:
             probs[i] = 1
         else:
             probs[i] = 0
             
     self.auc = roc_auc_score(y_test[:, 1], probs)
     print('AUC - Test Set: %.2f%%' % (self.auc*100))
    
     # calculate roc curve
     fpr, tpr, thresholds = roc_curve(y_test[:, 1], probs)
     # plot no skill
     plt.plot([0, 1], [0, 1], linestyle='--')
     # plot the roc curve for the model
     plt.plot(fpr, tpr, marker='.')
     plt.xlabel('False positive rate')
     plt.ylabel('Sensitivity/ Recall')
     # show the plot
     plt.show()
     
     self.precision = precision_score(y_test[:, 1], probs)
     print('Precision: %f' % self.precision)
     # recall: tp / (tp + fn)
     self.recall = recall_score(y_test[:, 1], probs)
     print('Recall: %f' % self.recall)
     # f1: tp / (tp + fp + fn)
     self.f1 = f1_score(y_test[:, 1], probs)
     print('F1 score: %f' % self.f1)

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		#plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		#plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
    
def save_performance(self):
    columnas = ['Date', 'Model', "epochs", "batch_train", "batch_test", "subjects", "Sessions", "fMRI_Modality", 'AUC', 'precision','recall','f1','path_weigths']					

    
    data1 = [[datetime.now(), self.model, self.epochs,self.batch_train,self.batch_test,self.subjects, self.sessions, self.fMRI_Modality, self.auc, self.precision,self.recall,self.f1,'path saving']]
  
    df1 = pd.DataFrame(data1, columns=columnas)
    
    path = "performance.csv"
    df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

    