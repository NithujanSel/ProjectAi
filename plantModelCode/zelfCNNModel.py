#Nodige libirary in laden
import os 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation, Flatten,Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications 
import datetime


#De trian data set in laden en dat verdelen tussen train en validatie set
train_path = "./farmLabDataSet"
IMAGE_SIZE = 224

#train set
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_path,
                                                                 label_mode="categorical", 
                                                                 shuffle=True,
                                                                 image_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                batch_size = 32,
                                                                 seed=42,
                                                                validation_split=0.2,
                                                                   subset="training")

#validatie set
valid_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_path,
                                                                 label_mode="categorical", 
                                                                 shuffle=False,
                                                                 image_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                batch_size = 32,
                                                                 seed=42,
                                                                validation_split=0.2,
                                                                   subset="validation")

#Model maken voor onze dataset

class_names = train_data.class_names #De fase naam opslaan.

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))  #Dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))



# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# import visualkeras
# from PIL import ImageFont
# font = ImageFont.truetype("arial.ttf", 35)
# visualkeras.layered_view(model,legend=True, to_file='output3.png',font=font)
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer = tf.keras.optimizers.Adam(), 
             metrics=['accuracy']) 

#Model runnen
model_history = model.fit(train_data, 
                          steps_per_epoch=len(train_data),
                          validation_data= valid_data,
                        validation_steps= len(valid_data),
                         epochs=1,
                         #callbacks=[tensorboard_callback]
                         )


#Model opslaan
model.save("./zelfGemaakteModel.h5")



















#https://www.kaggle.com/code/atrisaxena/using-tensorflow-2-x-classify-plant-seedlings
