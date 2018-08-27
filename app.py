#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
#from flask_socketio import SocketIO

import numpy as np 
import cv2
import time

from multiprocessing import Process, Lock
mutex= Lock()
stream_mutex = Lock()
last_frame = None
img_right = np.zeros(shape=(160, 160, 3))

#import websocket
app = Flask(__name__)
#socketio = SocketIO(app)
#from views import index

#from flask_socketio import send, emit
import json

#from websockets import (
#      handle_client_connect_event,
#)

#### https://tutorials.technology/tutorials/61-Create-an-application-with-websockets-and-flask.html
"""
@socketio.on('client_connected')
def handle_client_connect_event(json):
    #global 
    print('received jsonx: {0}'.format(str(json)))

@socketio.on('message')
def handle_json_button(json):
    print('[*] handle_json_button',json)
    # it will forward the json to all clients.
    with mutex:
        socketio.send(last_frame, json=True)
"""

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('pong.html')


def add_crops(img,bounding_boxes):
    for i,bb in enumerate(bounding_boxes):
        [x1,y1,x2,y2] = bounding_boxes[i]['bbox']
        # crop
        crop = img[y1:y2,x1:x2]
        if crop is not None and crop.shape[0]>0 and crop.shape[1]>0:
            # resize to 160x160 
            if crop.shape[0] < 160: ## shrinking
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            #print('before',crop.shape)
            crop = cv2.resize(crop,(160,160), interpolation = interpolation )
            bounding_boxes[i]['crop']=crop
    return bounding_boxes

import predict,db_extractor
face_expression_extractor = predict.FaceExpressionExtractor()
face_det = predict.FaceDetector()

## webcam
cap = cv2.VideoCapture(0)
try:
    from queue import LifoQueue
except:
    from Queue import LifoQueue
last_frames_left = LifoQueue(10)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def gen_right():
    global img_right
    while True:
        time.sleep(1. / 35.)
        with stream_mutex:
            encoded = cv2.imencode('.jpg', img_right)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def gen():
    global last_frame, img_right
    """Video streaming generator function."""
    while True:
        try:
            _, im = cap.read()
            faces = face_det.detect_faces(im,max_num=2)
            bounding_boxes = add_crops(im,faces)
            face_expressions = face_expression_extractor.get_face_expression_vectors(bounding_boxes)
            #print('faces',[f['bbox'] for f in faces])
            #print(face_expressions)
            ## combine face crops
            face_crops = [f['crop'] for f in bounding_boxes]
            c = np.zeros((160,320,3),np.uint8)
            if len(bounding_boxes) > 0:
                img_left = np.fliplr(bounding_boxes[0]['crop'])
            else:
                img_left = np.zeros(shape=(160, 160, 3))
            """
            if last_frames_left.full() == 10:
                last_frames_left.pop()
                last_frames_left.put(img_left)
                #img_left = moving_average(img_left, 10)
            else:
                last_frames_left.put(img_left)
            """
            
            with stream_mutex:
                if len(bounding_boxes) > 1:
                    img_right = np.fliplr(bounding_boxes[1]['crop'])
                else:
                    img_right = np.zeros(shape=(160, 160, 3))

            #s = 3
            #c = cv2.resize(c,None,fx=s, fy=s, interpolation = cv2.INTER_CUBIC)
            encoded = cv2.imencode('.jpg', img_left)[1].tobytes()
            #face_im = np.hstack(face1,face2)

            with mutex:
                last_frame = face_expressions


            try:
                #with Flask.test_request_context(app) as trc:
                data = face_expressions.reshape((-1,7)).tolist()
                #socketio.send(data, json=True)
            except Exception as e:
                print(e)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        except Exception as e:
            print(e)
#websocket 
#    send emotion_vectors

@app.route('/video_feed_left')
def video_feed_left():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_right')
def video_feed_right():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_right(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/expressions")
def expressions():
    global last_frame
    with mutex:
        if last_frame is not None:
            return json.dumps(last_frame.reshape((-1,7)).tolist())
            #return np.array2string(last_frame)
        else:
            return ""
        last_frame = None


"""
import asyncio
import websockets

async def hello(websocket, path):
    name = await websocket.recv()
    print('openend',name)
    #print("< {name}")

    #greeting = f"Hello {name}!"

    await websocket.send('greeting')
    print('sent succ')

start_server = websockets.serve(hello, 'localhost', 6000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
"""

if __name__ == '__main__':

    
    
    app.run(host='0.0.0.0', threaded=True)