from __future__ import print_function
import matplotlib 
matplotlib.use('Agg')

import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse

import time 

import tornado.ioloop
import tornado.web
import tornado.websocket

from vrnow.ar.server import generic_server
#from playground.face.expression import predict
import predict
face_expression_extractor = predict.FaceExpressionExtractor()
face_det = predict.FaceDetector()


code_directory = os.path.expanduser('~/face_pong')


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("pong.html")
        print('got http get request')


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

class PongWebsocketStreamer(tornado.websocket.WebSocketHandler):
    def open(self):
        self.frame_count = 0
        self.last_output_image = np.zeros((320,480),np.uint8)
        print("WebSocket opened")
        running = True
        ## webcam
        cap = cv2.VideoCapture(0)

        print('[*] hello webcam')
        while running:
            #try:
                ret, im = cap.read()
                print('image',im.shape,im.dtype,im.min(),im.max())
                faces = face_det.detect_faces(im,max_num=2)
                
                print(self.frame_count,'faces',[f['bbox'] for f in faces])
                bounding_boxes = add_crops(im,faces)
                print('bounding_boxes',bounding_boxes)
                
                

                # send data to client via websocket
                message = json.dump({'bb':bounding_boxes})

                self.frame_count += 1
            #except:
            #    pass
class PongWebsocketStreamerX(generic_server.ImageStreamingWebSocket):
    def imageTransformation(self,image_input):
        print('[*] imageTransformation new frame')
        if self.frame_count % 10 > 0: 
            return self.last_output_image

        image = np.array( image_input )
        #faces = face_det.detect_faces(image,max_num=2)
        
        #print(step,'faces',[f['bbox'] for f in faces])
        #bounding_boxes = add_crops(im,faces)
        try:
            expression_vectors = face_expression_extractor.getFaceExpressionVectors(image)
        
            if expression_vectors.shape[0] > 0:
                #print(fn.split('/')[-1],expression_vectors.shape)
                for (x,y,w,h),expression_vector in zip(face_expression_extractor.faces, expression_vectors):
                    # draw bounding box
                    color = (0,0,255)
        
                    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                    
                    # draw background rectangle for text
                    cv2.rectangle(image,(x+w+20,y),(x+w+100,y+h),(200,220,200),-1)

                    font = cv2.FONT_HERSHEY_PLAIN#cv2.FONT_HERSHEY_SIMPLEX
                    font_size = 0.75
                    for i,emotion_activation in enumerate(expression_vector):
                        #cv2.putText(image,face_expression_extractor.emotions[i] + ':' + str(round(emotion_activation,2)),(x+w+30, y + int(round(float(h)* float(i)/7.0)) + 35), font, 1.0,(0,0,0),2,cv2.LINE_AA)
                        cv2.putText(image,face_expression_extractor.emotions[i] + ':' + str(round(emotion_activation,2)),(x+w+22, y + int(round(float(h)* float(i)/7.0)) + 12), font, font_size,(220,70,95),1,cv2.LINE_AA)
                        emotion = face_expression_extractor.emotions[i]
        except:
            pass
        return image

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/websocket", PongWebsocketStreamer),
        (r'/js/(.*)', tornado.web.StaticFileHandler, {'path': os.path.join(code_directory,'js')}),
        (r'/images/(.*)', tornado.web.StaticFileHandler, {'path': os.path.join(code_directory,'images')}),
        (r'/sounds/(.*)', tornado.web.StaticFileHandler, {'path': os.path.join(code_directory,'sounds')})
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print('[*] game started')
    tornado.ioloop.IOLoop.current().start()
    