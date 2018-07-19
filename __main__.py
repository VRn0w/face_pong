from __future__ import print_function

import numpy as np 
import cv2
import tensorflow as tf

import predict,db_extractor
face_expression_extractor = predict.FaceExpressionExtractor()
face_det = predict.FaceDetector()

## webcam
cap = cv2.VideoCapture(0)
#PONG pygame

import random
import pygame, sys
from pygame.locals import *

pygame.init()
fps = pygame.time.Clock()

#colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

#globals
SIDE_WIDTH = 200
CANVAS_XOFFSET = SIDE_WIDTH
WIDTH = 600 #+ 2*SIDE_WIDTH
HEIGHT = 400       
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 120#80
HALF_PAD_WIDTH = PAD_WIDTH / 2
HALF_PAD_HEIGHT = PAD_HEIGHT / 2
ball_pos = [0,0]
ball_vel = [0,0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0

#canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('VRnow Face Emotion Pong')

# helper function that spawns a ball, returns a position vector and a velocity vector
# if right is True, spawn to the right, else spawn to the left
def ball_init(right):
    global ball_pos, ball_vel # these are vectors stored as lists
    ball_pos = [WIDTH/2,HEIGHT/2]
    _s = 2
    horz = random.randrange(2*_s,4*_s)
    vert = random.randrange(1*_s,3*_s)
    
    if right == False:
        horz = - horz
        
    ball_vel = [horz,-vert]

# define event handlers
def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel,l_score,r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH - 1,HEIGHT/2]
    paddle2_pos = [WIDTH +1 - HALF_PAD_WIDTH,HEIGHT/2]
    l_score = 0
    r_score = 0
    if random.randrange(0,2) == 0:
        ball_init(True)
    else:
        ball_init(False)


#draw function of canvas
def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score
           
    canvas.fill(BLACK)
    pygame.draw.line(canvas, WHITE, [WIDTH / 2, 0],[WIDTH / 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)

    # update paddle's vertical position, keep paddle on the screen
    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel
    
    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel

    #update ball
    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    #draw paddles and ball
    pygame.draw.circle(canvas, RED, ball_pos, 20, 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)

    #ball collision check on top and bottom walls
    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    
    #ball collison check on gutters or paddles
    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(paddle1_pos[1] - HALF_PAD_HEIGHT,paddle1_pos[1] + HALF_PAD_HEIGHT,1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)
        
    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(paddle2_pos[1] - HALF_PAD_HEIGHT,paddle2_pos[1] + HALF_PAD_HEIGHT,1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)

    #update scores
    myfont1 = pygame.font.SysFont("Roboto", 20)
    label1 = myfont1.render("Score "+str(l_score), 1, (255,255,0))
    canvas.blit(label1, (50,20))

    myfont2 = pygame.font.SysFont("Roboto", 20)
    label2 = myfont2.render("Score "+str(r_score), 1, (255,255,0))
    canvas.blit(label2, (470, 20))  
    
    
#keydown handler
def keydown(event):
    global paddle1_vel, paddle2_vel
    
    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_w:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8

#keyup handler
def keyup(event):
    global paddle1_vel, paddle2_vel
    
    if event.key in (K_w, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0

init()

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
#game loop
step = 0
while True:

    draw(window)
    ret, im = cap.read()
    faces = face_det.detect_faces(im,max_num=2)
    
    #print(step,'faces',[f['bbox'] for f in faces])
    bounding_boxes = add_crops(im,faces)
    try:
        UP_CLASSES = [3]
        DOWN_CLASSES = [1,2,5,4,0]
        dvel = 10
        thresh = 0.7
        face_expressions = face_expression_extractor.get_face_expression_vectors(bounding_boxes)
        paddle1_vel,paddle2_vel = 0,0
        #print('   ---')
        if len(face_expressions) > 0:
            max_expression_idx0 = int(np.argmax(face_expressions[0]))
            max_expression_val0 = face_expressions[0][max_expression_idx0]
            for class_id in UP_CLASSES: 
                if max_expression_idx0 == class_id and max_expression_val0>thresh:
                    paddle1_vel = -dvel
                    print('player0 up')
            for class_id in DOWN_CLASSES: 
                if max_expression_idx0 == class_id and max_expression_val0>thresh:
                    paddle1_vel = dvel
                    print('player0 down')
            #print('played player 0')

        ## second player available?
        if len(face_expressions) > 1:
            max_expression_idx1 = int(np.argmax(face_expressions[1]))
            max_expression_val1 = face_expressions[1][max_expression_idx1]
            for class_id in DOWN_CLASSES: 
                if max_expression_idx1 == class_id and max_expression_val0>thresh:
                    paddle2_vel = -dvel
                    print('player1 down')
            for class_id in UP_CLASSES: 
                if max_expression_idx1 == class_id and max_expression_val0>thresh:
                    paddle2_vel = dvel
                    print('player1 up')
            #print('played player 1')
        ## let bot move second paddle
        else:
            paddle2_pos[1] += ball_pos[1]-paddle2_pos[1]
        def draw_emotion_vectors():
            _font = pygame.font(SysFont('Roboto'),15)

        #myfont2 = pygame.font.SysFont("Roboto", 20)
        #labels_emotions = myfont2.render("Score "+str(r_score), 1, (255,255,0))
        #window.blit(label2, (470, 20))  
    
    
        print(step,'face_expressions',db_extractor.emotions[max_expression_idx0],face_expressions[0][max_expression_idx0])


    except Exception as e :
        print(e)
    step += 1
    for event in pygame.event.get():

        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()
            
    pygame.display.update()
    fps.tick(60)