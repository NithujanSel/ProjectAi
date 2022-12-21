from flask import Flask,render_template,Response
import re
import os
import cv2
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

app=Flask(__name__)
#Camera
camera=cv2.VideoCapture(1)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# AI model
# the TFLite converted to be used with edgetpu

# Verander de path van je eigen model
modelPath = '/home/mendel/AiProject/ModelLoc/ResNet/ResNetmodel_tflite_edgetpu.tflite'
# Verander de path van je eigen labels
labelPath = '/home/mendel/AiProject/ModelLoc/ResNet/ResNetmodel_labels.txt'

modelOutput = "Empty"
modelLabel = "label"
modelScore = 10

def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)


def generate_frames():
    global modelOutput
    global modelLabel 
    global modelScore 
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)
    while camera.isOpened():
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            # Classify and display image
            results = classifyImage(interpreter, frame)
            cv2.imshow('frame', frame)
            #modelScore  = labels[results[0].id]
            #modelScore = results[0].score
            modelOutput = f'Fase: {labels[results[0].id]} <br> Accuracy: {round(results[0].score * 100,2)}%'
            #print(modelOutput)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('start.html')

@app.route('/updatePerdict')
def updatePerdict():
    return modelOutput

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
    app.run(debug=True)
    
