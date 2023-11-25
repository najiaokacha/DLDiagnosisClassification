import io
import os
import sys

import pika
import ssl
#from receive import main
from path import PATH,MQ




def main():
    
    context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
    ssl_options = pika.SSLOptions(context, MQ)
    #connection = pika.BlockingConnection(pika.ConnectionParameters(ssl_options=ssl_options, port=5671, host=MQ, credentials=pika.PlainCredentials("kino","1593578915935789")))
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost" ))

    channel = connection.channel()
    channel.queue_declare(queue='rpc_queue')
    def on_request(ch, method, props, body):


        print("pong...")
       
        ch.basic_publish(exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id = \
                                                            props.correlation_id),
                        body='pong')
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    print(' [*] Waiting for messages. To exit press CTRL+C')



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

