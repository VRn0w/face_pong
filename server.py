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
from playground.face.expression import predict
face_expression_extractor = predict.FaceExpressionExtractor()


class FaceExpressionWebsocketStreamer(generic_server.ImageStreamingWebSocket):
    def imageTransformation(self,image_input):

        if self.frame_count % 10 > 0: 
            return self.last_output_image

        image = np.array( image_input )
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
        
        return image

def make_app():
    return tornado.web.Application([
        (r"/", generic_server.ImageMainHandler),
        (r"/websocket", FaceExpressionWebsocketStreamer)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
    print 'started FasterRCNN server'