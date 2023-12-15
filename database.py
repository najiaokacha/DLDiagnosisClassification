import io
import os
import ssl
import sys
import cv2
import numpy as np
import pika
from PIL import Image
from keras.models import load_model
from keras.optimizers import Adam

#from receive import main

# load model
custom_objects = {'Adam': Adam}  # Replace 'Adam' with the actual optimizer used during training

modelPnemonia = load_model('pnemonia.h5', compile=False)
model = load_model('brain-tumor-model.h5')
modelRetino = load_model('retino_detection.h5')
modelBreast = load_model('final_CNN.h5')
modelKeratocone=load_model('model_1.h5')

def names(number):
    if(number == 0):
        return "no_tumor"
    elif(number == 1):
        return "meningioma_tumor"
    elif(number == 2):
        return "pituitary_tumor"
    elif(number == 3):
        return "glioma_tumor"


def retinas(number):
    if(number == 0):
        return "Normal"
    elif(number == 1):
        return "Mild"
    elif(number == 2):
        return "Moderate"
    elif(number == 3):
        return "Severe"
    elif(number == 4):
        return "Proliferative"

def pneumonia_names(number):
    if (number == 0):
        return "Normal"
    elif (number == 1):
        return "Pneumonia"

def breast_names(number):
    if (number == 0):
        return "Normal"
    elif (number == 1):
        return "Breast Cancer"

def keratocone_names(number):
    if(number == 0):
        return "keratocone"
    elif(number == 1):
        return "normal"
    elif(number == 2):
        return "suspicious"


def riddleRetina(n):
    oracle = ""
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)
        img = Image.open(binary_data)
    except:
        oracle = "not an image"

    try:
        x = np.array(img.resize((224, 224)))
        x = x.reshape(1, 224, 224, 3)
        answ = modelRetino.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + retinas(classification)
    except:
        oracle = "bad image"
    return oracle


def riddle(n):
    oracle = ""
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)

        img = Image.open(binary_data)
    except:
        oracle = "not an image"

    try:
        x = np.array(img.resize((150, 150)))
        x = x.reshape(1, 150, 150, 3)
        answ = model.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + names(classification)
    except:
        oracle = "bad image"
    return oracle

def riddlePneumonia(n):
    oracle = ""
    try:
        # Open the image file
        binary_data = io.BytesIO(n)
        img = Image.open(binary_data)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize the image to the model's input size
        img = img / 255.0  # Normalize pixel values to the range [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(e)
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


def riddleBreast(n):
    oracle = ""

    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)
        img = Image.open(binary_data)
    except:
        oracle = "not an image"
        return oracle

    try:
        # Resize the image
        x = np.array(img.resize((50, 50)))
        x = x.reshape(1, 50, 50, 3)

        # Make predictions using the new model
        answ = modelBreast.predict_on_batch(x)
        print(answ)
        classification = np.where(answ == np.amax(answ))[1][0]

        oracle = f"{answ[0][classification] * 100:.2f}% Confidence This Is {breast_names(classification)}"
    except:
        oracle = "bad image"

    return oracle

def riddleKeratocone(file_path):
    oracle = ""
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(file_path)
        img = Image.open(binary_data)
    except Exception as e:
        oracle = f"Error processing image: {str(e)}"
        return oracle
    try:
        # Resize the image to match the expected input shape of the model
        resized_img = img.resize((128, 128))
        x = np.array(resized_img)
        x = x.reshape(1, 128, 128, 3)  # Ensure the image has 3 channels (RGB)
        x = x / 255.0

        # Make predictions using the new model
        answ = modelKeratocone.predict_on_batch(x)
        classification = np.argmax(answ)


        oracle = f"{answ[0][classification] * 100:.2f}% Confidence This Is {keratocone_names(classification)}"
    except Exception as e:
        oracle = f"Error processing image: {str(e)}"

    return oracle

context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
# ssl_options = pika.SSLOptions(context, MQ)
# connection = pika.BlockingConnection(pika.ConnectionParameters(
# ssl_options=ssl_options, port=5671, host=MQ, credentials=pika.PlainCredentials("kino", "**************")))
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_brain')
channel.queue_declare(queue='rpc_retino')
channel.queue_declare(queue='rpc_pneumonia')
channel.queue_declare(queue='rpc_breast')
channel.queue_declare(queue='rpc_keratocone')


def brain(ch, method, props, body):
    #  n = int.from_bytes(body, "big")
    print("classifying image...")
    response = riddle(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)


def retino(ch, method, props, body):

    print("classifying image...")
    response = riddleRetina(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)

def pneumonia(ch, method, props, body):

    print("classifying image...")
    response = riddlePneumonia(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)

def breast(ch, method, props, body):

    print("classifying image...")
    response = riddleBreast(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)

def keratocone(ch, method, props, body):

    print("classifying image...")
    response = riddleKeratocone(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rpc_brain', on_message_callback=brain)
    channel.basic_consume(queue='rpc_retino', on_message_callback=retino)
    channel.basic_consume(queue='rpc_pneumonia', on_message_callback=pneumonia)
    channel.basic_consume(queue='rpc_breast', on_message_callback=breast)
    channel.basic_consume(queue='rpc_keratocone', on_message_callback=keratocone)
    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    print(' [*] Waiting for messages. To exit press CTRL+C')


if _name_ == '_main_':

    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
