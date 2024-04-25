# -*- coding: utf-8 -*-
from load_data import get_frames, get_label_names
from predict_instability import predict_instability, load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'my_model_vial1_vial3_with_blur.h5'


def print_accuracy(test_x, test_y, label_names, model_path = MODEL_PATH):

    predicted_y = []
    
    # testing if the network does good predictions
    load_model(model_path)
    for frame in test_x:
        frame = frame.reshape(-1, frame.shape[0], frame.shape[1], 1)
        
        (prediction, predictions) = predict_instability(frame)
        predicted_y.append(prediction)
        
    conf_matrix = confusion_matrix(test_y, predicted_y)
    
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")
    
    precision, recall, fbeta_score, support = precision_recall_fscore_support(test_y, predicted_y)
    label_idx = np.unique(test_y)
    print("precision:")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(precision)
    print("recall:")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(recall)
    print("fbeta_score:")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(fbeta_score)
    print("support:")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(support)
    print("false positive rate (Fall out):")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(FPR)
    print("false negative rate:")
    for i in label_idx: print(label_names[i], end='\t\t')
    print(" ")
    print(FNR)
    
    
    
    
    