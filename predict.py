"""
this script predicts images for facial expression 


python playground/face/expression/predict.py -input_file /home/deeplearning/data/deepdream/input/ada/a364bfbb-bfcb-404b-b310-bd4452a81dbe.jpg
    
"""

from __future__ import print_function

from glob import glob

import numpy as np 
import cv2
import tensorflow as tf 
import dlib

#import face_detection
import model, db_extractor
import sys,os
sys.path.append(os.path.expanduser('~'))
from ai.applied.face.facenet.src.align import detect_face

#   setup facenet parameters
gpu_memory_fraction = 0.4
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


class FaceDetector(object):
    def __init__(self, backend="dlib"):
        self.backend = backend
        print("using backend %s" % backend)
        if backend == "facenet":
            self.load_facenet_model()
        elif backend == "dlib":
            self.detector = dlib.get_frontal_face_detector()

    def load_facenet_model(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess_detect = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                log_device_placement=False))
            self.sess_detect = sess_detect
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn( self.sess_detect, None)

    def detect_faces(self, img, max_num):
        if self.backend == "dlib":
            return self.detect_faces_dlib(img, max_num)
        elif self.backend == "facenet":
            return self.detect_faces_facenet(img, max_num)
        else:
            raise Exception("Could not find backend %s" % self.backend)

    def detect_faces_dlib(self, img, max_num):
        results = []
        detected = self.detector(img, 0)
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            box = [x1, y1, x2, y2]
            faces = self.get_faces_from_bounding_box(img, box)
            results.append(faces)
        return self.get_middle_faces(img, results, max_num)

    ### get most middle faces
    def get_middle_faces(self, img, faces, num):
        if len(faces) == 0: return []

        xcenter = img.shape[1]//2
    
        min_idx0,min_idx1 = 0,0
        min_dist = abs( faces[0]['bbox'][0]-xcenter )
        for i in range(1,len(faces)):
            dd = abs( faces[i]['bbox'][0]-xcenter )
            if dd < min_dist:
                min_idx1 = int(min_idx0)
                min_idx0 = i 
                min_dist = dd

        middle_faces = [faces[min_idx0]]
        if num == 2 and len(faces)>1:
            middle_faces.append(faces[min_idx1])
            # make face0 left from face1
            if middle_faces[0]['bbox'][0]<middle_faces[1]['bbox'][0]:
                _t = middle_faces[0]
                middle_faces[0] = middle_faces[1]
                middle_faces[1] = _t

        return middle_faces


    def get_faces_from_bounding_box(self, img, box, padding=30):
        # make square
        [x1,y1,x2,y2] = box
        w = x2-x1
        h = y2-y1
        dif = w - h 
        if dif > 0: # higher than wider
            x1-=dif//2
            x2+=dif//2
        else:
            y1-=dif//2
            y2+=dif//2

        # padding for bigger roi
        p = 30
        x1 -= p
        y1 -= p
        x2 += p
        y2 += p

        # crop
        results = []
        crop = img[y1:y2,x1:x2]
        if crop is not None and crop.shape[0]>0 and crop.shape[1]>0:
            # resize to 160x160 
            if crop.shape[0] < 160: ## shrinking
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            #print('before',crop.shape)
            crop = cv2.resize(crop,(160,160), interpolation = interpolation )
            
            return {'bbox': [x1,y1,x2,y2], 'crop':crop}
        else:
            return None

    def detect_faces_facenet(self, img, max_num):
        #   run detect_face from the facenet library
        bounding_boxes, _ = detect_face.detect_face(
                img, minsize, self.pnet,
                self.rnet, self.onet, threshold, factor)
        result = []
        #   for each box
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            box = np.around([x1,y1,x2,y2]).astype(np.int32)
            box_results = self.get_faces_from_bounding_box(img, box)
            box_results.update({"acc": acc})
            result.append(box_results)
            
        #result = result[:min(len(result),max_num)]             
        result = self.get_middle_faces(img, result, max_num)
        return result

class FaceExpressionExtractor(object):
    def __init__(self):
        if True:
            self.loadModel()
        #except:
        #    print("FaceExpressionExtractor ERROR: couldn't load model!")
        self.emotions = db_extractor.emotions
        self.step = 0

    def loadModel(self):

        self.images = tf.placeholder(tf.float32, [None]+list(model.IMAGE_SHAPE))        
        self.logits = model.inference(self.images, keep_prob = 0.5)
        self.softmax = tf.nn.softmax(self.logits)

        # Create a saver.
        saver = tf.train.Saver(model.get_inference_variables(),write_version=tf.train.SaverDef.V2)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(init)
    
        # try to load trained model 
        ckpt = tf.train.get_checkpoint_state(model.checkpoint_directory)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,ckpt.model_checkpoint_path)
            print('restored model from '+str(ckpt.model_checkpoint_path))

    def get_face_expression_vectors(self,bounding_boxes):
        self.step += 1
        
        batch = []
        for i, bb in enumerate(bounding_boxes):
            x,y = bb['bbox'][:2]
            roi_face = bb['crop']
            # make gray
            roi_gray = cv2.cvtColor(roi_face, cv2.COLOR_RGB2GRAY)

            # resize to 48x48
            patch = cv2.resize(roi_gray,tuple(model.IMAGE_SHAPE[:2]), interpolation = cv2.INTER_AREA)
            cv2.imwrite('/tmp/face/%i_%i.jpg'%(self.step,i),patch)
            batch.append(patch)
        
        # reshape and normalize batch       
        batch = np.array(batch,np.float32) / 128. - 1.
        batch = np.reshape(batch, [len(bounding_boxes)] + list(model.IMAGE_SHAPE) )

        [softmax] = self.sess.run([self.softmax],feed_dict={self.images: batch})
        return softmax

if __name__ == '__main__':
    from playground.ops import parseArguments
    args = parseArguments.getParsedArgs()
    if args.output_file is None:
        try:
            args.output_file = '/data/face/detection/' + args.input_file.split('/')[-1]
        except:
            pass

    if args.input_file is None and args.output_file is None:
        fns = glob('/home/deeplearning/data/deepdream/input/ada/*')
    else:
        fns = [args.input_file]

    color = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    face_expression_extractor = FaceExpressionExtractor()

    for fn in fns:
        image = cv2.imread(fn)

        expression_vectors = face_expression_extractor.getFaceExpressionVectors(image)
        if expression_vectors.shape[0] > 0:
            print(fn.split('/')[-1],expression_vectors.shape)
            for (x,y,w,h),expression_vector in zip(face_expression_extractor.faces, expression_vectors):
                # draw bounding box
                cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                
                # draw background rectangle for text
                cv2.rectangle(image,(x+w+20,y),(x+w+220,y+h),(255,255,255),-1)

                for i,emotion_activation in enumerate(expression_vector):
                    cv2.putText(image,face_expression_extractor.emotions[i] + ':' + str(round(emotion_activation,2)),(x+w+30, y + int(round(float(h)* float(i)/7.0)) + 35), font, 1.0,(0,0,0),2,cv2.LINE_AA)
                    emotion = face_expression_extractor.emotions[i]
                    print(emotion,':',emotion_activation)
                print('')

        if expression_vectors.shape[0] > 0:
            cv2.imwrite('/data/face/expression/output_images/'+fn.split('/')[-1],image)



            





