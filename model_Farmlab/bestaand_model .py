import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications

import numpy as np

from tensorflow.keras.applications import MobileNet,ResNet50,EfficientNetB0,InceptionV3

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet import preprocess_input
import datetime
# the parameters
IMAGE_SIZE = 224
EPOCHS= 2
train_path = "./newDataSet"
model_naam = "ResNetmodel"

#included in our dependencies
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(train_path,
                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical', shuffle=True)

print(len(train_generator.class_indices))
print(train_generator.class_indices)

with open(f'{model_naam}_labels.txt', 'w') as my_file:
        for i in train_generator.class_indices.keys():
            my_file.write(f'{i}\n')
            
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array= img_array/255.0
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)



def build_keras_model():
  base_model=MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
    include_top = False, weights = "imagenet", classes = 1000 )

  for layer in base_model.layers:
    layer.trainable = False
    
  # the last few layers
  # for the flowers dataset, we are predicting 5 types of flower which is reason for the final layer to have 5 outputs.

  x=base_model.output
  x=GlobalAveragePooling2D()(x)
  x=Dense(100,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
  x=Dropout(0.5)(x)
  x=Dense(50,activation='relu')(x) #dense layer 50
  preds=Dense(len(train_generator.class_indices),activation='softmax')(x) #final layer with softmax activation  
  model=Model(inputs=base_model.input,outputs=preds)
  for layer in model.layers[86:]:
    layer.trainable=True    
  return model




#train
step_size_train=train_generator.n//train_generator.batch_size
model = build_keras_model()

# import visualkeras
# from PIL import ImageFont
# font = ImageFont.truetype("arial.ttf", 128)
# visualkeras.layered_view(model,legend=True, to_file='InceptionV3.png',font=font)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_generator,steps_per_epoch=step_size_train,epochs=EPOCHS)#,callbacks=[tensorboard_callback])

model.save(f'{model_naam}')


def representative_dataset_gen():
  for i in range(0,5):
    x,y=train_generator.next()
    image=x[i:i+1]
    yield [image]
    
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
  print('tfresult', tf_results)
  print('tflite', tflite_result)

# Convert to tflite model
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(f'{model_naam}')
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(f'{model_naam}_tflite.tflite', 'wb') as o_:
    o_.write(tflite_model)      
