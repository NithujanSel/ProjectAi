from keras.models import load_model #Om de ML model te laden
import numpy as np #om beter met array te werken
import cv2 #Om fotos te lezen en manuplueren
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
IMAGE_SIZE=224

model = load_model("./ResNetmodel") #model in laden
foto = "./f2.png"#De foto inladen om de model te testen
label_path = "./ResNetmodel_labels.txt"

#Gaat de foto om zetten naar 224x244 grotte.
def fotoResize(PATHF):
    img_Array = cv2.imread(PATHF)
    img_Array = cv2.resize(img_Array,(IMAGE_SIZE,IMAGE_SIZE))
    img_Array = np.expand_dims(img_Array, axis=0)

    return img_Array



foftX = fotoResize(foto) #Uitvoering van fotoResize functie

foftX = np.array(foftX) #Omzetten naar np array


print(foftX.shape) #Kijken of de foto goed is


# perdict = np.argmax(model.predict(foftX), axis=-1)
x = model.predict([foftX])[0] #de foto perdicten welke fase het is
print(x)
perdict = np.argmax(x,axis=-1) #Heeft een array aleen de juiste fase pakken
print(perdict)
with open("./ResNetmodel_labels.txt",'r') as fp:
    for i, line in enumerate(fp):
        if i == perdict:
            print(line)
            break




#print(list(klasse.keys())[list(klasse.values()).index(perdict-1)])
