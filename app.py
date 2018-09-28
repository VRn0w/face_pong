#!/usr/bin/env python
from importlib import import_module
import os, shutil
from flask import Flask, render_template, Response, request
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
        ta = img.shape[0]*img.shape[1]
        perc_im = (float(ta) - (w*h)) / float(ta)
        #perc_im = 1. - perc_im
        perc_im = round(perc_im,1)
        #print('perc_im',perc_im)
        m = int(30 * perc_im * 4.5)
        #m = int(0.4 * (x2-x1))
        
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
face_det = predict.FaceDetector(no_face_after = 1)

## webcam
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)
VIDEO_WIDTH,VIDEO_HEIGHT = 640,480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,VIDEO_HEIGHT)
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
            #print('[*] found %d faces.' % len(faces))
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
                        if len(bounding_boxes)>1 and face_expressions[1][i] > 0.3:
                            if queues_right[i].full(): queues_right[i].get(0)
                            queues_right[i].put((1. - face_expressions[1][i],img_right))


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

"""
@app.route("/print_best_of")
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
    shutil.copyfile(file_collage,os.path.expanduser('~/face_pong/static/bestof.png'))
    print('[*] saved collage to disk')
    return file_collage
"""
@app.route("/print_best_of")
def print_best_of():
    global queues_left,queues_right

    # make collage of most expressive face crops
    file_collage = os.path.join(collage_dir,'%d.png'%int(np.random.uniform() * 1e5))

    p = 2 
    logo0, logo1 = [ cv2.resize(cv2.imread(os.path.join(image_dir,f)),(160,160)) for f in ['logo_vrnow.jpg','logo_nhs.jpg'] ]
    #collage = np.vstack((logo0, 255*np.ones((p,160,3),np.uint8), logo1))
    collage = np.vstack((logo0, logo1))

    # find maximum number
    lefts,rights = [],[]
    for i in range(6):
        if not queues_left[i].empty():
            act, p = queues_left[i].get(0)
            lefts.append(p) 
        if not queues_right[i].empty():
            act, p = queues_right[i].get(0)
            rights.append(p) 
    n = min(len(lefts),len(rights))
    print('lengths',len(lefts),len(rights),'N =',n)
    ## 1 player
    if len(rights) == 0:
        if len(lefts)%2==1:
            lefts = lefts[1:]
        for i in range(0,len(lefts),2):
            print(i,'lefts',lefts[i].shape,lefts[i+1].shape,collage.shape)   
            
            #x = np.vstack((lefts[i],255 * np.ones((p,160,3),np.uint8),lefts[i+1]))
            #collage = np.hstack((collage, 255* np.ones((2*160+p,p,3),np.uint8) ,x))

            #neue_spalte = np.vstack((lefts[i],255*np.ones((p,160,3),np.uint8),lefts[i+1]))
            neue_spalte = np.vstack((lefts[i],lefts[i+1]))
            collage = np.hstack((collage,neue_spalte))
    else:
        for i in range(n):
            #collage = np.hstack((collage,255*np.ones((collage.shape[0],p,3),np.uint8),np.vstack((lefts[i],255*np.ones((p,160,3),np.uint8),rights[i]))))
            collage = np.hstack((collage,np.vstack((lefts[i],rights[i]))))
    
    cv2.imwrite(file_collage,collage)
    shutil.copyfile(file_collage,os.path.expanduser('~/face_pong/static/bestof.png'))
    print('[*] saved collage to disk')

    queues_left = [PriorityQueue(maxsize=max_face_crops_collage) for _ in range(6)]
    queues_right = [PriorityQueue(maxsize=max_face_crops_collage) for _ in range(6)]
    return file_collage


from sender import send_email

"""
@app.route("/screenshot")
def screenshot():
    import gtk.gdk

    w = gtk.gdk.get_default_root_window()
    sz = w.get_size()
    if sz[0] > 1920:
        width, height = sz[0] // 2, sz[1]
    else:
        width, height = sz
    print("The size of the window is %d x %d" % sz)
    pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
    pb = pb.get_from_drawable(w, w.get_colormap(),0,0,0,0,sz[0],sz[1])
    pb= pb.subpixbuf(0, 0, width, height)
    if (pb != None):
        pb.save("static/screenshot.png", "png")
        return json.dumps({"status": "success"})
        # return image_to_base64("screenshot.png")
        print("Screenshot saved to screenshot.png.")
    else:
        abort(500)
        print("Unable to get the screenshot.")
        return ""
"""

@app.route('/email', methods=['POST'])
def email_route():
    image_path = os.path.expanduser("~/face_pong/static/bestof.png")
    name = request.values.get('name', None)
    email = request.values.get('email', None)
    company = request.values.get('company', None)
    if name == "" or email == "" or company ==  "" or not "@" in email:
        abort(500)
    else:
        if send_email(name, email, company, image_path):
            return json.dumps({"status": "success"})
        else:
            abort(400)

if __name__ == '__main__':

    if True:
        import logging
        log = logging.getLogger('werkzeug')    
        log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', threaded=True,port=5000)

