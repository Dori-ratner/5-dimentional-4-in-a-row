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
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization,Reshape,Conv2D
from keras.models import Model
#tf.compat.v1.enable_eager_execution()



import math
import env 
import random
import keras
pygame.init()

BLUE = (0,0,255)
SQURESIZE = 125
NUM_COL = 7
NUM_ROW = 6
RADIUS = SQURESIZE/2 -3
width = NUM_COL * SQURESIZE
hight = NUM_ROW * SQURESIZE
size = (width,hight)


a = keras.initializers.RandomUniform(minval=0, maxval=0.05, seed=0)

# Define Q-network

#= Sequential([
#shape((6, 7, 1), input_shape=(42,)),  # Reshape input to (6, 7, 1)
#nv2D(filters=8, kernel_size=(4, 4), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#atten(),
#nse(7, activation="linear", kernel_initializer=a, bias_initializer='zeros', name='dense_4')
#
#
#
#_net = Sequential([
#shape((6, 7, 1), input_shape=(42,)),  # Reshape input to (6, 7, 1)
#nv2D(filters=8, kernel_size=(4, 4), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#nv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"),
#atten(),
#nse(7, activation="linear", kernel_initializer=a, bias_initializer='zeros', name='dense_4')
#

Q_net = Sequential([
Dense(512,input_shape = (42,), activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(256, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(7, activation="linear", kernel_initializer= a ,bias_initializer='zeros')])

Target_net = Sequential([
Dense(512,input_shape = (42,), activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(256, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(128, activation="relu", kernel_initializer= a ,bias_initializer='zeros'),
Dense(7, activation="linear", kernel_initializer= a ,bias_initializer='zeros')])

ALPHA=0.1
# Compile the Q-network with a suitable loss function and optimizer
Q_net.compile(optimizer=Adam(learning_rate=ALPHA), loss="mse")

fresh_play = False

#get the pre trained model from my drive
if (fresh_play == True):
     link = r"G:\האחסון שלי\dqn_10000_wheights.h5"
else:
     link = r"C:5 dimentional 4 in a row\model_register\Q_net.weights.h5"

Q_net.load_weights(link)
Q_net.save_weights(r'C:\Users\dorir\OneDrive\Desktop\5 dimentional 4 in a row\model_register\Q_net.weights.h5')
Target_net.set_weights(Q_net.get_weights())



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

def animate_game(column,screen, x, player,env):
    x = x.copy()
    t = np.where(x[:, column] == 0)
    if t[0].size == 0:
        print("Tried to insert full for player:", player)
    else:
        row_to_drop = t[0][-1]  # Get the lowest empty row in the column
        x[row_to_drop, column] = player
        pygame.time.wait(100)
        for r in range(NUM_ROW):
            if r <= row_to_drop:
                y_position = r * SQURESIZE + SQURESIZE // 2
                color = (100, 0, 0) if player == 1 else (0, 100, 0)
                pygame.draw.circle(screen, color, (column * SQURESIZE + SQURESIZE // 2, y_position), RADIUS)
                pygame.display.update()
                pygame.time.delay(25)  # Adjust this delay to control the falling speed

                draw_board(screen)
                draw_game(screen,env)

            else:
                # Redraw the background over the disk
                draw_board(screen)
                pygame.display.update()
                break  # Stop drawing once the disk reaches its final position
     
               

screen = pygame.display.set_mode(size)
draw_board(screen)
pygame.display.update()

Env = env.env((NUM_ROW,NUM_COL))

ai_turn = random.choice([True, False]) # who starts
states = []
actions = []
#game loop:_________________________________________
game_over = False
while not game_over:
    states.append(Env.get_state())
    draw_game(screen,Env)
    pygame.display.update()
    if ai_turn:
        #ai turn--
               b = False
               action = Env.can_win_block(1)# i win
               if(action == -2):
                    action = Env.can_win_block(-1)# he wins then block  fixxxxxxxxxx
                    if(action == -2):
                            b = True
                            q_values = Q_net.predict(Env.get_state().reshape(-1,42))
                            print(q_values)
                            q_values[0][np.where(Env.get_state()[0,:] !=0)] = -1 #takes only first row as size is same
                            animate_game(np.argmax(q_values),screen,x=Env.get_state(),player=1,env=Env)
                            Env.insert(np.argmax(q_values),1)
               if(not b):
                    animate_game(action,screen,x=Env.get_state(),player=1,env=Env)
                    Env.insert(action,1)# insert the best action

               draw_game(screen,Env)
               pygame.display.update()
               ai_turn = False


               if Env.wincheck()!=-2:
                    pygame.display.update()
                    pygame.time.wait(5000)#10 seconds delay
                    g = np.array(Env.get_state())
                    g[g==1] = 4
                    g[g==-1]= 1
                    print(g)
                    if Env.wincheck()==0: print('tie')
                    if Env.wincheck()==1: print('red won')
                    else: print('green won')
                    game_over = True

    clicked = False
    while not clicked:
     for event in pygame.event.get():
          if event.type == pygame.QUIT: 
               pygame.quit() 
               pygame.display.quit()
               
          if event.type == pygame.MOUSEBUTTONDOWN:
               clicked = True


          
               if Env.wincheck()!=-2:
                    
                    pygame.time.wait(5000)#10 seconds delay
                    g = np.array(Env.get_state())
                    g[g==1] = 4
                    g[g==-1]= 1
                    print(g)
                    if Env.wincheck()==0: print('tie')
                    if Env.wincheck()==1: print('red won')
                    else: print('green won')
                    game_over = True

               #my turn--
               posx = event.pos[0]
               colum = int(math.floor(posx/SQURESIZE))
               animate_game(colum,screen,x=Env.get_state(),player=-1,env=Env)
               actions.append(Env.insert(colum,-1))
               draw_game(screen,Env)
               pygame.display.update()

               if Env.wincheck()!=-2:
                    
                    pygame.time.wait(5000)#10 seconds delay
                    g = np.array(Env.get_state())
                    g[g==1] = 4
                    g[g==-1]= 1
                    print(g)
                    if Env.wincheck()==0: print('tie')
                    if Env.wincheck()==1: print('red won')
                    else: print('green won')
                    game_over = True

               ai_turn = True



reply_buffer = []

rew = Env.wincheck()
if(len(states)-1 == len(actions) and len(states) < 42):actions.append(-1)# add last action if bot won in less then 9 moves
for l in range(0,len(actions)-1):
    reply_buffer.append((states[l].reshape(42), states[l+1].reshape(42), rew  , actions[l]))#reward part of the BELLMAN EQU

batch = reply_buffer
states,next_states ,rew , actions = zip(*batch)
#----------------
states = np.array(states)
next_states = np.array(next_states)
rew = np.array(rew)
actions  = np.array(actions)
#----------------
Q_values = Q_net.predict(states,verbose = 0)
next_Q = Target_net.predict(next_states,verbose = 0)
for i in range(len(states)):
    if(i!=len(states)-1):#calaulate reward
        Q_values[i,actions[i]] = rew[i] + 0.99 * np.max(next_Q[i]) #bellman equeation
    else:
        Q_values[i,actions[i]] = rew[i]
#train the model
history = Q_net.fit(states,Q_values, epochs = 100,verbose =1)
Target_net.set_weights(Q_net.get_weights())# move whights every 10 epocs

Q_net.save_weights(r'5 dimentional 4 in a row\model_register\Q_net.weights.h5')
