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

collage_dir = '/data/collage/'
if not os.path.isdir(collage_dir): os.makedirs(collage_dir)

image_dir = os.path.expanduser('~/face_pong/static/images')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('pong.html')


def add_crops(img,bounding_boxes):
    for i,bb in enumerate(bounding_boxes):
        [x1,y1,x2,y2] = bounding_boxes[i]['bbox']
        

        # crop
        crop = img[y1:y2,x1:x2]
        ## to get from celebA to emotion dataset, crop the center
        h, w = y2-y1,x2-x1
        m = int(0.2 * (x2-x1))
        crop = crop[m:-m,m:-m]
        
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
    from queue import LifoQueue, PriorityQueue
except:
    from Queue import LifoQueue, PriorityQueue
last_frames_left = LifoQueue(10)

max_face_crops_collage = 2
queues_left = [PriorityQueue(maxsize=max_face_crops_collage) for _ in range(6)]
queues_right = [PriorityQueue(maxsize=max_face_crops_collage) for _ in range(6)]

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
    global last_frame, img_right, queues_left,queues_right
    """Video streaming generator function."""
    while True:
        time.sleep(1. / 35.)
        try:
            _, im = cap.read()
            faces = face_det.detect_faces(im,max_num=2)
            #print('[*] found %d faces' % len(faces))
            bounding_boxes = add_crops(im,faces)
            face_expressions = face_expression_extractor.get_face_expression_vectors(bounding_boxes)

            ## combine face crops
            face_crops = [f['crop'] for f in bounding_boxes]
            c = np.zeros((160,320,3),np.uint8)
            if len(bounding_boxes) > 0:
                img_left = np.fliplr(bounding_boxes[0]['crop'])
            else:
                img_left = np.zeros(shape=(160, 160, 3))
            
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
                last_frame = {'face_expressions':face_expressions}
                if len(bounding_boxes) > 0:# and False:
                    for i in range(6): # dont add neutral faces
                        if face_expressions[0][i] > 0.3:
                            if queues_left[i].full(): queues_left[i].get(0)
                            queues_left[i].put((1. - face_expressions[0][i],img_left))


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        except Exception as e:
            #print(e)
            ''


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
            return json.dumps(last_frame['face_expressions'].reshape((-1,7)).tolist())
            #return np.array2string(last_frame)
        else:
            return ""
        last_frame = None

@app.route("/print_bestof")
def print_best_of():
    # make collage of most expressive face crops
    file_collage = os.path.join(collage_dir,'%d.jpg'%int(np.random.uniform() * 1e5))

    collage = 255 * np.ones((2,160,3),np.uint8)
    for i in range(6):
        if not queues_left[i].empty():
            act,p = queues_left[i].get(0)
            collage = np.vstack((collage,p,255*np.ones((2,160,3),np.uint8)))

    ## add logos
    if True:
        logo0 = cv2.imread(os.path.join(image_dir,'logo_vrnow.jpg'))
        logo1 = cv2.imread(os.path.join(image_dir,'logo_nhs.jpg'))
        logo0 = cv2.resize(logo0,(160,160))
        collage = np.vstack((collage,np.zeros((2,160,3),np.uint8),logo0))
    cv2.imwrite(file_collage,collage)
    print('[*] saved collage to disk')
    return file_collage

if __name__ == '__main__':

    if True:
        import logging
        log = logging.getLogger('werkzeug')    
        log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', threaded=True)

