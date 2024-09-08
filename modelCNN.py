from keras import *
from keras.layers import *
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.applications import ResNet101



def buildModel():

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224, 3))
    newLayer = base_model.output
    newLayer = GlobalAveragePooling2D()(newLayer)
    newLayer = Dense(512, activation='relu')(newLayer)
    newLayer = Dropout(0.5)(newLayer)
    predictions = Dense(1, activation='sigmoid')(newLayer)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    optimizer = optimizers.SGD(lr=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  

    return model

model = buildModel()
model.summary()