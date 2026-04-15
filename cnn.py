import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
aug_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    "data/train", target_size=(224,224),
    batch_size=32, class_mode='categorical', subset='training'
)

val_data = train_gen.flow_from_directory(
    "data/train", target_size=(224,224),
    batch_size=32, class_mode='categorical', subset='validation'
)

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "data/test", target_size=(224,224),
    batch_size=32, class_mode='categorical', shuffle=False
)

model_base = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(3,activation='softmax')
])

model_base.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history_base = model_base.fit(train_data,validation_data=val_data,epochs=5)

model_reg = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3,activation='softmax')
])

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model_reg.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

class_weights = {0:1.0,1:2.0,2:3.0}

train_data_aug = aug_gen.flow_from_directory(
    "data/train", target_size=(224,224),
    batch_size=32, class_mode='categorical', subset='training'
)

val_data_aug = aug_gen.flow_from_directory(
    "data/train", target_size=(224,224),
    batch_size=32, class_mode='categorical', subset='validation'
)

history_reg = model_reg.fit(
    train_data_aug,
    validation_data=val_data_aug,
    epochs=5,
    class_weight=class_weights
)

base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128,activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(3,activation='softmax')(x)

model_resnet = tf.keras.Model(inputs=base_model.input,outputs=output)

model_resnet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_resnet.fit(train_data_aug,validation_data=val_data_aug,epochs=5)

y_pred = model_resnet.predict(test_data)
y_pred_labels = np.argmax(y_pred,axis=1)
y_true = test_data.classes

print(confusion_matrix(y_true,y_pred_labels))
print(classification_report(y_true,y_pred_labels))