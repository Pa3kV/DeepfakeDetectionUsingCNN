import sys
import tensorflow as tf
import json
import os
import cv2
import pandas as pd
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.image as mpimg

def showAucRoc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC vrijednost = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('AUC-ROC evaluacija')
    plt.legend(loc='lower right')
    plt.show()

framesRoot = os.path.join(os.getcwd() + '/frames/test')

modelPath = os.path.join(os.getcwd() + '/results/Models/DeepfakeDetection_8.h5')

model = models.load_model(modelPath)

historyPath = os.path.join(os.getcwd() + '/results/Histories/training_history_8.json') 

with open(historyPath, 'r') as file:
    history = json.load(file)

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

testGenerator = test_datagen.flow_from_directory(
    framesRoot,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False 
)

# test_loss, test_accuracy = model.evaluate(testGenerator)
# print(f'Test Accuracy: {test_accuracy:.4f}')
# print(f'Test Loss: {test_loss:.4f}')

y_pred = model.predict(testGenerator)
y_pred = (y_pred > 0.5).astype(int).ravel()

y_true = testGenerator.classes

falsePositives = np.where((y_pred == 0) & (y_true == 1))[0]
falseNegatives = np.where((y_pred == 1) & (y_true == 0))[0]
filenames = testGenerator.filenames

print(classification_report(y_true, y_pred, target_names=testGenerator.class_indices.keys()))
print(confusion_matrix(y_true, y_pred))
showAucRoc(y_true, y_pred)

for id in falsePositives:
    imgPath = os.path.join(os.getcwd(), "frames", "test", testGenerator.filepaths[id])
    savePath = os.path.join(os.getcwd(), "results", "False Results", "False Positives", filenames[id].split('\\')[1])
    shutil.copy(imgPath, savePath)

for id in falseNegatives:
    imgPath = os.path.join(os.getcwd(), "frames", "test", testGenerator.filepaths[id])
    savePath = os.path.join(os.getcwd(), "results", "False Results", "False Negatives", filenames[id].split('\\')[1])
    shutil.copy(imgPath, savePath)