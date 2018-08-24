from __future__ import print_function
""" https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/ """

import csv
import os 

import cv2
import numpy as np 

directory_fer2013 = '/data/datasets/face/expression/kaggle/fer2013/'
fn_csv = directory_fer2013 + 'fer2013.csv'

emotions = { 0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

def printStats():
    with open(fn_csv,'rb') as csvfile:
        reader = csv.reader(csvfile)
        """
            row hat 3 Elemente
            Usage ['emotion', 'pixels', 'Usage']
            row[0] integer [0,6] Klasse (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
            row[1] image data
            row[2] dataset PublicTest 3589 Training 28709 PrivateTest 3589

        """
        labels = {}
        emotions_count = np.zeros((7,),np.int32)
        for i,row in enumerate(reader):
            if i == 0: print(row)
            if i>0:
                if row[2] not in labels:
                    labels[row[2]] = 1
                else:
                    labels[row[2]] += 1

                emotions_count[int(row[0])] +=1

        for i in range(7):
            print(emotions[i],'\t',emotions_count[i]/float(emotions_count.sum()))

        """
        ['emotion', 'pixels', 'Usage']

        labels:
        Usage 1
        PublicTest 3589
        Training 28709
        PrivateTest 3589

        Angry    0.138016551955
        Disgust  0.0152422882938
        Fear     0.142697912893
        Happy    0.250480675454
        Sad      0.169337085853
        Surprise 0.111516705214
        Neutral  0.172708780338

        """

def saveImages(output_directory = directory_fer2013 + 'images/', dtype = 'png'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(fn_csv,'rb') as csvfile:
        reader = csv.reader(csvfile)
        for i , (emotion, pixels_string, usage ) in enumerate(reader):
            if i > 0:
                fn_image = output_directory + usage + '_' + str(i) + '_' + emotions[int(emotion)]   + '.' + dtype
                image = np.array(pixels_string.split(' '),dtype = np.uint8).reshape((48,48))
                cv2.imwrite(fn_image, image)
                print(i, 'saved image to',fn_image)

if __name__ == '__main__':
    
    printStats()

    writeImages = True
    if writeImages:
        saveImages()
    else:
        print('not writing images!')
        