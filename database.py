import json
import io
import os
import re
import ssl
import sys
import cv2
import numpy as np
import pika
from PIL import Image
from keras.models import load_model
from keras.optimizers import Adam

# Chemin vers le répertoire contenant les fichiers h5
directory_path = r"C:\Users\admin\Documents\projet_maladies\Brain-cancer"

# Variable pour stocker les informations
models_info = []
# Parcours des fichiers du répertoire
for filename in os.listdir(directory_path):
    if filename.endswith(".h5"):
        # Utiliser une regex pour extraire les informations du nom de fichier
        match = re.match(r'([\w\s]+)_(\w+-\w+-\w+-\w+-\w+)\.h5', filename)
        if match:
            maladie_name, uuid = match.groups()
            model_info = {
                'maladie_name': maladie_name,
                'uuid': uuid,
                'file_name': filename
            }
            models_info.append(model_info)
loaded_models = {}
for model_info in models_info:
    model_name = model_info.get('maladie_name')
    model_path = os.path.join(directory_path, model_info.get('file_name'))

    if model_name == "pneumonia":  # Use '==' for comparison
        loaded_model = load_model(model_path, compile=False)
    else:
        loaded_model = load_model(model_path)

    loaded_models[model_name] = loaded_model
def dynamic_riddle(n, maladie_name, width, heigth, normalize):
    oracle = ""
    try:
        binary_data = io.BytesIO(n)
        img = Image.open(binary_data)

        if maladie_name == "pneumonia":
            # Apply specific processing for pneumonia
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, heigth))
            img = img / normalize
            img = np.expand_dims(img, axis=0)
        else:
            # Default processing for other diseases
            img = np.array(img.resize((width, heigth)))
            img = img.reshape(1, width, heigth, 3)
            img = img / normalize

    except:
        oracle = "Error processing image"
        print(oracle)

    try:
        # Load the dynamic model based on the maladie_name
        model = loaded_models.get(maladie_name)

        if model:
            answ = model.predict_on_batch(img)
            classification = np.where(answ == np.amax(answ))[1][0]
            oracle = f"{answ[0][classification] * 100:.2f}% Confidence This Is {classification}"
        else:
            oracle = f"Model not available for disease: {maladie_name}"
    except:
        oracle = "Error predicting disease"

    return oracle


context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
# ssl_options = pika.SSLOptions(context, MQ)
# connection = pika.BlockingConnection(pika.ConnectionParameters(
# ssl_options=ssl_options, port=5671, host=MQ, credentials=pika.PlainCredentials("kino", "******")))
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


def dynamic_riddle_handler(ch, method, props, body):
    try:
        # Find the index of the JSON closing brace ('}')
        json_end_index = body.find(b'}') + 1

        # Decode the JSON part
        json_part = body[:json_end_index].decode('utf-8')
        payload = json.loads(json_part)

        # Extract information from the payload
        width = payload.get('width')
        height = payload.get('height')
        normalization_value = payload.get('normalizationValue')
        photo_data = body[json_end_index:]

        maladie_name = payload.get('nom')

        if maladie_name:
            print(f"Classifying image for Desease: {maladie_name}...")

            response = dynamic_riddle(photo_data, maladie_name, width, height, normalization_value)

            print(response)
            ch.basic_publish(exchange='',
                             routing_key=props.reply_to,
                             properties=pika.BasicProperties(
                                 correlation_id=props.correlation_id),
                             body=str(response))

            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            print("Desease name not found in payload.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # Handle JSON decoding error, if necessary


def dynamic_main():
    channel.basic_qos(prefetch_count=1)

    # Assuming queues are named dynamically based on Desease names
    for maladie_name in loaded_models.keys():
        queue_name = 'rpc_' + maladie_name.lower()
        channel.queue_declare(queue=queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=dynamic_riddle_handler)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    print(' [*] Waiting for messages. To exit press CTRL+C')


if __name__ == '__main__':

    try:
        dynamic_main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
