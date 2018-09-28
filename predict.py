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
from collections import deque

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
    def __init__(self, backend="dlib", padding=30, resize_size=(160,160), no_face_after=10):
        self.backend = backend
        self.padding = padding
        self.resize_size = resize_size
        print("using backend %s" % backend)
        if backend == "facenet":
            self.load_facenet_model()
        elif backend == "dlib":
            self.detector = dlib.get_frontal_face_detector()
        self.face_crops = deque(maxlen=10)
        self.no_face_after = no_face_after # when no face is detected take last frame and return no face after no_face_after frames
        self.last_faces = []
        self.no_faces_counter = no_face_after

    def load_facenet_model(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess_detect = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                log_device_placement=False))
            self.sess_detect = sess_detect
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn( self.sess_detect, None)

    def detect_faces(self, img, max_num, crop=None):
        
        y,x = img.shape[:2]
        if crop is not None:
            startx = x//2-(crop[0]//2)
            starty = y//2-(crop[1]//2)
            cropped_img = img[starty:starty+crop[1],startx:startx+crop[0], :]
        else:
            cropped_img = img

        if self.backend == "dlib":
            boxes = self.detect_faces_dlib(cropped_img, max_num, img.shape)
        elif self.backend == "facenet":
            boxes = self.detect_faces_facenet(cropped_img, max_num, img.shape)
        else:
            raise Exception("Could not find backend %s" % self.backend)
        
        boxes = self.sort_faces(img, boxes, max_num)


        if crop is not None:
            for i, bbox in enumerate(boxes):
                bbox[0] += startx
                bbox[2] += startx
                bbox[1] += starty
                bbox[3] += starty
                boxes[i] = bbox
            
        return [self.get_crop(bbox, img) for bbox in boxes]

    def detect_faces_dlib(self, img, max_num, shape):
        results = []
        detected = self.detector(img, 0)
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            box = [x1, y1, x2, y2]
            face = self.get_face_from_bounding_box(img, box, shape)
            results.append(face)
        return results

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

    def move_with_offset(self, x1, y1, x2, y2, offset=100):
        x1_n = (x1 - offset) if  (x1 - offset) > 0 else 0
        y1_n = (y1 - offset) if  (y1 - offset) > 0 else 0
        x1_diff = x1 - x1_n
        y1_diff = y1 - y1_n
        y2 = y2 - y1_diff
        x2 = x2 - x1_diff
        return x1_n, y1_n, x2, y2

    def add_padding(self, x1, y1, x2, y2, p=100):
        x1 -= p
        y1 -= p
        x2 += p
        y2 += p
        return x1, y1, x2, y2

    def make_square(self, box, shape):
        [x1, y1, x2, y2] = box
        w = x2-x1
        h = y2-y1
        dif = w - h 
        if dif > 0: # higher than wider
            x1-=dif//2
            x2+=dif//2
        else:            # wait a sec until training starts

            y1-=dif//2
            y2+=dif//2
        # print('before: ', [x1, y1, x2, y2], "max:", shape)
        if y2 > shape[0]:
            diff = y2 - shape[0]
            y2 = shape[0]
            x2 -= diff
        if x2 > shape[1]:
            diff = x2 - shape[1]
            x2 = shape[1]
            y2 -= diff
        return [x1, y1, x2, y2]

    def get_face_from_bounding_box(self, img, box, shape, basic_ratio=0.70):
        # make square
        [x1,y1,x2,y2] = box        
        w = x2-x1
        h = y2-y1
        [x1,y1,x2,y2] = self.make_square(box, shape=shape)

        # padding for bigger roi
        p_face = ((w / float(h)) - basic_ratio) * shape[1] / 1.8
        p_face = int(p_face)
        total_area = float(shape[0] * shape[1])
        percentage_of_image =  (total_area - (w * h)) / total_area
        percentage_of_image = 1.0 - percentage_of_image 
        p = int(self.padding * percentage_of_image * 4.5) + p_face
        o = int(30 * percentage_of_image * 6)
        # print("adding padding %d and offset %d" % (p, o))
        x1, y1, x2, y2 = self.add_padding(x1, y1, x2, y2, p)
        #x1, y1, x2, y2 = self.move_with_offset(x1, y1, x2, y2, o)

        self.face_crops.appendleft((x1, y1, x2, y2))

        '''
        if len(self.face_crops) == self.face_crops.maxlen:
            crops = np.asarray(self.face_crops)
            crop = np.mean(crops, axis=0)
            crop = crop.astype(np.int32)
            x1, y1, x2, y2 = list(crop)
            # print("Get the mean %s" % crop)
        '''    
        return self.make_square([x1, y1, x2, y2], shape)
    
    def get_crop(self, bbox, img):
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2,x1:x2]
        if crop is not None and crop.shape[0]>0 and crop.shape[1]>0:
            # resize to 160x160 
            if crop.shape[0] < self.resize_size[0]: ## shrinking
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            #print('before',crop.shape)
            crop = cv2.resize(crop, self.resize_size, interpolation = interpolation )
            
            return {'bbox': [x1,y1,x2,y2], 'crop':crop}
        else:
            return None
        

    def detect_faces_facenet(self, img, max_num, shape):
        #   run detect_face from the facenet library
        bounding_boxes, _ = detect_face.detect_face(
                img, minsize, self.pnet,
                self.rnet, self.onet, threshold, factor)

        if self.no_faces_counter > 0 and len(bounding_boxes) != max_num:
            self.no_faces_counter -= 1
            bounding_boxes = self.last_faces
        else:
            self.last_faces = bounding_boxes
            self.no_faces_counter = self.no_face_after

        result = []
        #   for each box
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            box = np.around([x1,y1,x2,y2]).astype(np.int32)
            box_results = self.get_face_from_bounding_box(img, box, shape)
            result.append(box_results)
            
        #result = result[:min(len(result),max_num)]             
        # result = self.get_middle_faces(img, result, max_num)
        #result = self.sort_faces(img, result, max_num)

        # returns only the boxes
        return result
    

    def sort_faces(self, img, boxes, max_num):
        xcenter, ycenter = int(img.shape[1] / 2.0), int(img.shape[0] / 2.0)
        boxes_center = [ int((b[0] + b[2]) / 2.)  for b in boxes]
        
        #boxes_center = [[ int((b[0] + b[2]) / 2.) , int((b[1] + b[3]) / 2.) ] for b in boxes]
        #boxes_diff = [abs(center[0] - xcenter) + abs(center[1] - ycenter) for center in boxes_center]
        #idxs = np.argsort(boxes_diff)
        idxs = np.argsort(boxes_center)

        return reversed([boxes[i] for i in idxs][:max_num])

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



            





