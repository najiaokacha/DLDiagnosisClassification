import cv2
from PIL import Image
import io
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import pika
import ssl
import traceback
# from receive import main
from path import MQ

# load model
model = load_model('brain-tumor-model.h5')
modelRetino = load_model('retino_detection.h5')
modelPnemonia = load_model('pnemonia.h5', compile=False)
modelBreast = load_model('final_CNN.h5')


def names(number):
    if (number == 0):
        return "glioma_tumor"
    elif (number == 1):
        return "meningioma_tumor"
    elif (number == 2):
        return "no_tumor"
    elif (number == 3):
        return "pituitary_tumor"


def retinas(number):
    if (number == 0):
        return 'Normal'
    elif (number == 1):
        return 'Mild'
    elif (number == 2):
        return 'Moderate'
    elif (number == 3):
        return 'Severe'
    elif (number == 4):
        return 'Proliferative'

def pneumonia_names(number):
    if (number == 0):
        return 'Normal'
    elif (number == 1):
        return 'Pneumonia'

def breast_names(number):
    if (number == 0):
        return 'Normal'
    elif (number == 1):
        return 'Breast Cancer'


def riddle(file_path):
    oracle = ""
    try:
        # Open the image file
        img = Image.open(file_path)
    except:
        oracle = "Failed to read image from file system"
        return oracle

    try:
        x = np.array(img.resize((150, 150)), dtype='uint8')

        # x = np.array(img.resize((150, 150)))
        x = x.reshape(1, 150, 150, 3)
        answ = model.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        print("Shape of x:", x.shape)
        print("Data type of x:", x.dtype)
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + names(classification)
    except Exception as e:
        oracle = "Bad image, error: " + str(e)

    return oracle


def riddleRetina(file_path):
    oracle = ""
    try:
        # Open the image file
        img = Image.open(file_path)
    except:
        oracle = "Failed to read image from file system"
        return oracle

    try:
        x = np.array(img.resize((224, 224)))
        x = x.reshape(1, 224, 224, 3)
        answ = modelRetino.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + retinas(classification)
    except:
        oracle = "Bad image"

    return oracle

def pnemonia(file_path):
    oracle = ""
    try:
        # Open the image file
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize the image to the model's input size
        img = img / 255.0  # Normalize pixel values to the range [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    except:
        oracle = "Failed to read or process the image"
        return oracle

    try:
        # Make a prediction using the loaded model
        answ = modelPnemonia.predict(img)
        classification = np.argmax(answ, axis=1)[0]
        confidence = np.max(answ) * 100
        oracle = f"{confidence:.2f}% Confidence This Is {pneumonia_names(classification)}"
    except Exception as e:
        print(e)
        oracle = "Error during prediction"

    return oracle


def riddle_breast(file_path):
    oracle = ""
    try:
        # Open the image file
        img = Image.open(file_path)
    except:
        oracle = "Failed to read image from the file system"
        return oracle

    try:
        # Resize the image
        dsize = (50, 50)
        img = cv2.imread(file_path)
        img = cv2.resize(img, dsize)

        # Preprocess the image for the new model
        x = np.array(img, dtype='uint8')
        x = x.reshape(1, 50, 50, 3)

        # Make predictions using the new model
        answ = modelBreast.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]

        oracle = f"{answ[0][classification] * 100:.2f}% Confidence This Is {breast_names(classification)}"
    except Exception as e:
        oracle = "Bad image, error: " + str(e)

    return oracle

# Example usage
#result = pnemonia("C:/Users/admin/Documents/projet_maladies/Brain-cancer/chest_xray/test/NORMAL/NORMAL-4512-0001.jpeg")
#print(result)

print(riddle_breast("C:/Users/admin/Documents/projet_maladies/Brain-cancer/chest_xray/test/PNEUMONIA/BACTERIA-40699-0001.jpeg"))
