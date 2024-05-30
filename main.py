import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential #NO
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.losses import MeanSquaredError
from keras.layers import Flatten
from keras import layers
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model

#tf.compat.v1.enable_eager_execution()



import math
import env 
import random
import keras
pygame.init()

BLUE = (0,0,255)
SQURESIZE = 100
NUM_COL = 7
NUM_ROW = 6
RADIUS = SQURESIZE/2 -3
width = NUM_COL * SQURESIZE
hight = NUM_ROW * SQURESIZE
size = (width,hight)


#load model from drive\downlads
Q_net = tf.keras.models.load_model(r'C:\Users\dorir\Downloads\dqn.h5')

print(Q_net.get_weights())



CLOCK = pygame.time.Clock()


def draw_board(screen):
     for c in range(NUM_COL):
          for r in range(NUM_ROW):
               pygame.draw.rect(screen,BLUE,(c*SQURESIZE,r*SQURESIZE,SQURESIZE,SQURESIZE))
               pygame.draw.circle(screen,(1,2,1),(c*SQURESIZE+SQURESIZE/2,r*SQURESIZE+SQURESIZE/2),RADIUS)

def draw_game(screen,Env):
     x = Env.get_state()
     print()
     for c in range(NUM_COL):
          for r in range(NUM_ROW):
               if (x[r,c] == 1):
                    color = (100,0,0)
                    pygame.draw.circle(screen,color,(c*SQURESIZE+SQURESIZE/2,r*SQURESIZE+SQURESIZE/2),RADIUS)
               if (x[r,c] == -1):
                    color = (0,100,0)
                    pygame.draw.circle(screen,color,(c*SQURESIZE+SQURESIZE/2,r*SQURESIZE+SQURESIZE/2),RADIUS)
               


screen = pygame.display.set_mode(size)
draw_board(screen)
pygame.display.update()

Env = env.env((NUM_ROW,NUM_COL))



#game loop:_________________________________________
game_over = False
while not game_over:
    draw_game(screen,Env)
    pygame.display.update()

    clicked = False
    while not clicked:
     for event in pygame.event.get():
          if event.type == pygame.QUIT: 
               pygame.quit() 
               pygame.display.quit()
               
          if event.type == pygame.MOUSEBUTTONDOWN:
               clicked = True


               
               #my turn--
               posx = event.pos[0]
               colum = int(math.floor(posx/SQURESIZE))
               Env.insert(colum,-1)


               #ai turn--
               draw_game(screen,Env)
               pygame.display.update()
               b = False
               action = Env.can_win_block(1)# i win
               if(action == -2):
                    action = Env.can_win_block(-1)# he wins then block  fixxxxxxxxxx
                    if(action == -2):
                            b = True
                            q_values = Q_net.predict(Env.get_state().reshape(-1,42))
                            print(q_values)
                            q_values['dense_4'][np.where(Env.get_state()[0,:] !=0)] = -1 #takes only first row as size is same
                            Env.insert(np.argmax(q_values),1)
               if(not b):
                    Env.insert(action,1)# insert the best action
               draw_game(screen,Env)
               pygame.display.update()

               CLOCK.tick(1)



               if Env.wincheck()!=-2:
                    CLOCK.tick(0.5)
                    if Env.wincheck()==0: print('tie')
                    if Env.wincheck()==1: print('red won')
                    else: print('green won')
                    game_over = True
