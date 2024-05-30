import numpy as np
import random

class env:

    def __init__(self,size = (6,7)):
        self.map = np.zeros(size)
        self.size = size
    def reset(self,size):
        self.map = np.zeros(size)

    def insert(self,num,player):
        """num: betwen 0-6 ,player: either 1 or -1"""
        t = np.where(self.map[:,num] == 0)
        if(t[0].size == 0):
            print("tried to insert full for player: ")
            print(player)
        else:
            self.map[t[0][-1],num] = player
        return num


    def can_win_block(self,player):
        """returns the winning/bloking action"""
        for i in range(0,self.size[1]):
            x = self.map.copy()
            t = np.where(x[:,i] == 0)
            if( not t[0].size == 0):
                x[t[0][-1],i] = player
                if self.wincheck(x)==player: return i
        return -2


    def get_state(self):
        return self.map

    def random_player(self ,player):
        w = np.where(self.map[0,:] == 0) # all the empty places, first row
        if(len(w[0])==0):print('duck')
        action = w[0][random.randint(0,len(w[0])-1)]
        return self.insert(action,player)


    def better_win_check(self, dig):
        for i in range(0,len(dig)-3):
            if(sum(dig[i:i+4])== 4):
                return 1
            if(sum(dig[i:i+4])== -4):
                return -1
        return 0

    def wincheck(self,x = []):#generelised wincheck for every mapsize
        map = x
        if len(x)==0: map = self.map
        digs=[] #array to add smaller arrays to and check later
        size_x = map.shape[0] - 3
        size_y = map.shape[1] - 3
        for i in range(len(map[0])):
            if(i<6):
                digs.append(map[i,:].T)
            digs.append(map[:,i])
            if(i<size_x):
                digs.append(map.diagonal(i))
                digs.append(np.diag(np.fliplr(map),i))
            if(0<i<size_y):
                digs.append(map.diagonal(-i))
                digs.append(np.diag(np.fliplr(map),-i))
        for i in range(len(digs)):
            if(self.better_win_check(digs[i]) == 1):
                return 1
            if(self.better_win_check(digs[i]) == -1):
                return -1
        if(len(np.where(x==0))==0): return 0
        return -2

