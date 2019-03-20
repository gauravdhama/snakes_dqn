# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:21:38 2017

@author: e065689
"""

# Snake implementation for reinforcement learning
import random,sys
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.preprocessing.image import load_img,img_to_array,array_to_img
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
import numpy as np
from collections import deque
from pygame.locals import *
import pygame

global mypixels
class snakes:
    def __init__(self):
        self.WINDOWWIDTH = 320
        self.WINDOWHEIGHT = 240
        self.CELLSIZE = 20
        self.CELLWIDTH = int(self.WINDOWWIDTH / self.CELLSIZE)
        self.CELLHEIGHT = int(self.WINDOWHEIGHT / self.CELLSIZE)
        startx = 5
        starty = 5
        self.wormCoords = [{'x': startx,     'y': starty},
                           {'x': startx - 1, 'y': starty},
                           {'x': startx - 2, 'y': starty}]
        self.direction = 'RIGHT'
        self.apple = {'x': 3, 'y': 3}
        self.done = 0
        self.HEAD = 0
        self.score = 0
        self.actions = ['UP','DOWN','RIGHT','LEFT']
        self.state_size = (self.CELLWIDTH,self.CELLHEIGHT,3)
        self.action_size = len(self.actions)
        self.WHITE     = (255, 255, 255)
        self.BLACK     = (  0,   0,   0)
        self.RED       = (255,   0,   0)
        self.GREEN     = (  0, 255,   0)
        self.DARKGREEN = (  0, 155,   0)
        self.DARKGREY  = ( 40,  40,  40)
        self.state = np.zeros(self.state_size)
        if self.done!=1:
            for c in self.wormCoords:
                self.state[c['x'],c['y'],0]=self.DARKGREY[0]
                self.state[c['x'],c['y'],1]=self.DARKGREY[1]
                self.state[c['x'],c['y'],2]=self.DARKGREY[2]
            self.state[self.apple['x'],self.apple['y'],0] = self.DARKGREY[0]
            self.state[self.apple['x'],self.apple['y'],1] = self.DARKGREY[1]
            self.state[self.apple['x'],self.apple['y'],2] = self.DARKGREY[2]
    def step(self,event):
        if self.done == 1:
            return (self.state,self.score,self.done)
        else:
            if (event == 'LEFT') and self.direction not in ('RIGHT'):
                self.direction = 'LEFT'
            elif (event == 'RIGHT') and self.direction not in ('LEFT'):
                self.direction = 'RIGHT'
            elif (event == 'UP') and self.direction not in ('DOWN'):
                self.direction = 'UP'
            elif (event == 'DOWN') and self.direction not in ('UP'):
                self.direction = 'DOWN'
                
            if self.direction == 'UP':
                newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y']-1}
                self.score = self.score+1
            elif self.direction == 'DOWN':
                newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y']+1}
                self.score = self.score+1
            elif self.direction == 'LEFT':
                newHead = {'x': self.wormCoords[self.HEAD]['x']-1, 'y': self.wormCoords[self.HEAD]['y']}
                self.score = self.score+1
            elif self.direction == 'RIGHT':
                newHead = {'x': self.wormCoords[self.HEAD]['x']+1, 'y': self.wormCoords[self.HEAD]['y']}
                self.score = self.score+1
            self.wormCoords.insert(0, newHead)
            if self.wormCoords[self.HEAD]['x'] == -1 or self.wormCoords[self.HEAD]['x'] == self.CELLWIDTH or self.wormCoords[self.HEAD]['y'] == -1 or self.wormCoords[self.HEAD]['y'] == self.CELLHEIGHT:
                print ('Crashed into a wall')
                self.done = 1
            for wormBody in self.wormCoords[1:]:
                if wormBody['x'] == self.wormCoords[self.HEAD]['x'] and wormBody['y'] == self.wormCoords[self.HEAD]['y']:
                    self.done = 1
                    print ('Crashed Myself')
            if self.wormCoords[self.HEAD]['x'] == self.apple['x'] and self.wormCoords[self.HEAD]['y'] == self.apple['y']:
                self.apple = {'x': 3, 'y': 3}
                self.score = self.score+100
            else:
                del self.wormCoords[-1]
            self.update_state()
            return (self.state,self.score,self.done)
    def reset(self):
        self.WINDOWWIDTH = 320
        self.WINDOWHEIGHT = 240
        self.CELLSIZE = 20
        self.CELLWIDTH = int(self.WINDOWWIDTH / self.CELLSIZE)
        self.CELLHEIGHT = int(self.WINDOWHEIGHT / self.CELLSIZE)
        startx = 5
        starty = 5
        self.wormCoords = [{'x': startx,     'y': starty},
                           {'x': startx - 1, 'y': starty},
                           {'x': startx - 2, 'y': starty}]
        self.direction = 'RIGHT'
        self.apple = {'x': 3, 'y': 3}
        self.done = 0
        self.HEAD = 0
        self.score = 0
        self.actions = ['UP','DOWN','RIGHT','LEFT']
        self.state_size = (self.CELLWIDTH,self.CELLHEIGHT,3)
        self.action_size = len(self.actions)
        self.WHITE     = (255, 255, 255)
        self.BLACK     = (  0,   0,   0)
        self.RED       = (255,   0,   0)
        self.GREEN     = (  0, 255,   0)
        self.DARKGREEN = (  0, 155,   0)
        self.DARKGREY  = ( 40,  40,  40)
        self.state = np.zeros(self.state_size)
        if self.done!=1:
            for c in self.wormCoords:
                self.state[c['x'],c['y'],0]=self.DARKGREY[0]
                self.state[c['x'],c['y'],1]=self.DARKGREY[1]
                self.state[c['x'],c['y'],2]=self.DARKGREY[2]
            self.state[self.apple['x'],self.apple['y'],0] = self.DARKGREY[0]
            self.state[self.apple['x'],self.apple['y'],1] = self.DARKGREY[1]
            self.state[self.apple['x'],self.apple['y'],2] = self.DARKGREY[2]
    def get_state(self):
        return self.state
    def update_state(self):
        self.state = np.zeros(self.state_size)
        if self.done!=1:
            for c in self.wormCoords:
                self.state[c['x'],c['y']]=1
            self.state[self.apple['x'],self.apple['y']] = 1
    def drawWorm(self,wormCoords):
        for coord in wormCoords:
            x = coord['x'] * self.CELLSIZE
            y = coord['y'] * self.CELLSIZE
            wormSegmentRect = pygame.Rect(x, y, self.CELLSIZE, self.CELLSIZE)
            if wormCoords.index(coord)==0:
                pygame.draw.rect(self.DISPLAYSURF, self.WHITE, wormSegmentRect)
                wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, self.CELLSIZE - 8, self.CELLSIZE - 8)
                pygame.draw.rect(self.DISPLAYSURF, self.RED, wormInnerSegmentRect)
            else:
                pygame.draw.rect(self.DISPLAYSURF, self.DARKGREEN, wormSegmentRect)
                wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, self.CELLSIZE - 8, self.CELLSIZE - 8)
                pygame.draw.rect(self.DISPLAYSURF, self.GREEN, wormInnerSegmentRect)
    def drawApple(self,coord):
        x = coord['x'] * self.CELLSIZE
        y = coord['y'] * self.CELLSIZE
        appleRect = pygame.Rect(x, y, self.CELLSIZE, self.CELLSIZE)
        pygame.draw.rect(self.DISPLAYSURF, self.RED, appleRect)
    def drawGrid(self):
        for x in range(0, self.WINDOWWIDTH, self.CELLSIZE): # draw vertical lines
            pygame.draw.line(self.DISPLAYSURF, self.DARKGREY, (x, 0), (x, self.WINDOWHEIGHT))
        for y in range(0, self.WINDOWHEIGHT, self.CELLSIZE): # draw horizontal lines
            pygame.draw.line(self.DISPLAYSURF, self.DARKGREY, (0, y), (self.WINDOWWIDTH, y))
    def drawScore(self,score):
        scoreSurf = self.BASICFONT.render('Score: %s' % (score), True, self.WHITE)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.WINDOWWIDTH - 120, 10)
        self.DISPLAYSURF.blit(scoreSurf, scoreRect)
    def showGameOverScreen(self):
        gameOverFont = pygame.font.Font('freesansbold.ttf', 150)
        gameSurf = gameOverFont.render('Game', True, self.WHITE)
        overSurf = gameOverFont.render('Over', True, self.WHITE)
        gameRect = gameSurf.get_rect()
        overRect = overSurf.get_rect()
        gameRect.midtop = (self.WINDOWWIDTH / 2, 10)
        overRect.midtop = (self.WINDOWWIDTH / 2, gameRect.height + 10 + 25)
        self.DISPLAYSURF.blit(gameSurf, gameRect)
        self.DISPLAYSURF.blit(overSurf, overRect)
    def render(self):
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.FPS = 15
        self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
        self.BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        BGCOLOR = self.BLACK
        self.DISPLAYSURF.fill(BGCOLOR)
        #self.drawGrid()
        self.drawWorm(self.wormCoords)
        self.drawApple(self.apple)
        #self.drawScore(len(self.wormCoords) - 3)
        #self.pxarray = pygame.PixelArray(self.DISPLAYSURF)
        #del self.pxarray
        #if self.done==1:
        #    self.showGameOverScreen()
        #pygame.display.update()
        pygame.image.save(self.DISPLAYSURF,'current_state.jpg')
        #self.FPSCLOCK.tick(self.FPS)
    def terminate(self):
        pygame.quit()
        sys.exit()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(20, kernel_size=(3, 3), input_shape=(240,320,3),data_format="channels_last"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(30, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('relu'))
        
        model.add(Dropout(0.1))    
        
        model.add(Dense(4))
        model.add(Activation('sigmoid'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),metrics=['accuracy'])
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.random()<self.epsilon:
            return random.choice([0,1,2,3])
        else:
            act_values = self.model.predict(np.array([state]))
            return np.argmax(act_values[0])
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f,epochs=1,verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = snakes()
env.reset()

agent = DQNAgent(env.state_size,env.action_size)
episodes = 100

for e in range(episodes):
    # reset state in the beginning of each game
    env.reset()
    break_var = 1
    for time_t in range(100):
        env.render()
        state0 = load_img('current_state.jpg')
        state = img_to_array(state0)
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        next_state, reward, done = env.step(env.actions[action])
        env.render()
        next_state0 = load_img('current_state.jpg')
        next_state = img_to_array(next_state0)
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}, lifecycles : {}"
                  .format(e, episodes, reward,time_t))
            break_var = time_t
            break
    if break_var==0:
        break
    # train the agent with the experience of the episode
    agent.replay(time_t)