import json
import os
from keras import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import modelCNN 
from keras.callbacks import EarlyStopping

framesRoot = os.path.join(os.getcwd() + '/frames/train')

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    framesRoot,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    framesRoot,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


model = modelCNN.buildModel()

STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n // validation_generator.batch_size

earlyStopping = EarlyStopping(
    monitor='val_loss',   # Praćenje gubitka na validacijskom setu
    patience=5,           # Broj epoha čekanja prije zaustavljanja ako nema poboljšanja
    restore_best_weights=True  # Vraćanje težina iz najbolje epohe
)

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=15
)

modelPath = os.path.join(os.getcwd(), "results", "Models")

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

numOfFiles = len(os.listdir(modelPath))
fileName = f'DeepfakeDetection_{numOfFiles + 1}.h5'

model.save(os.path.join(modelPath, fileName))

historyPath = os.path.join(os.getcwd(), "results", "Histories")

if not os.path.exists(historyPath):
    os.makedirs(historyPath)

fileHisName = f'training_history_{numOfFiles + 1}.json'
fileHisPath = os.path.join(historyPath, fileHisName)

with open(fileHisPath, 'w') as f:
    json.dump(history.history, f)
