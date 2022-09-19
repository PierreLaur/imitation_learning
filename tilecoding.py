'''
A Tile Coding implementation for 2D environments
'''
import numpy as np

class TileCoding :
    def __init__(self,num_tilings,num_tiles_in_row,num_tiles_in_column,x_range,y_range,rbf=False) :
        self.tilings=[]
        self.num_tilings=num_tilings
        self.x_range=np.array(x_range,dtype='float32')
        self.y_range=np.array(y_range,dtype='float32')
        self.tile_width=(x_range[1]-x_range[0])/num_tiles_in_row
        self.tile_height=(y_range[1]-y_range[0])/num_tiles_in_column
        self.x_offset=self.tile_width/num_tilings
        self.y_offset=self.tile_height/num_tilings
        self.rbf=rbf
        self._add_tilings(num_tiles_in_row,num_tiles_in_column)

    def _add_tilings(self,num_tiles_in_row,num_tiles_in_column) :
        for t in range(self.num_tilings) :
            self.tilings.append(Tiling(self.tile_width,self.tile_height,self.x_offset*t,self.y_offset*t,self.x_range,self.y_range,num_tiles_in_row,num_tiles_in_column,rbf=self.rbf))

    def encode_state(self,state) :
        feat_vector=np.array([])
        for tiling in self.tilings :
            feat_vector=np.append(feat_vector,tiling.encode_state(state))
        return feat_vector

class Tiling :
    def __init__(self,tile_width,tile_height,x_offset,y_offset,x_range,y_range,num_tiles_in_row,num_tiles_in_column,rbf) :
        self.tile_width=tile_width
        self.tile_height=tile_height
        self.x_offset=x_offset
        self.y_offset=y_offset
        self.x_range=x_range
        self.y_range=y_range
        self.num_tiles_in_row=num_tiles_in_row
        self.num_tiles_in_column=num_tiles_in_column
        self.rbf=rbf

    def encode_state (self,state) :
        feats=np.zeros([self.num_tiles_in_row,self.num_tiles_in_column])
        
        if self.x_range[0] > state[0] or state[0] > self.x_range[1] or self.y_range[0] > state[1] or state[1] > self.y_range[1] :
            print("error : out of range")
            print(state)
            print(self.x_range[0] > state[0],type(self.x_range[0]),type(state[0]))
            print(-1.2>-1.2)
            print("ranges :",self.x_range,self.y_range)
            print("\n\n")
            return feats

        if (self.x_range[0]+self.x_offset <= state[0] <= self.x_range[1]+self.x_offset) and (self.y_range[0]+self.y_offset <= state[1] <= self.y_range[1]+self.y_offset) :
            normalized_x=(state[0]-self.x_range[0]-self.x_offset)
            normalized_y=(state[1]-self.y_range[0]-self.y_offset)
            active_x=int(normalized_x//self.tile_width)
            active_y=int(normalized_y//self.tile_height)
            if self.rbf==True :
                center=[self.x_offset+self.tile_width/2,self.y_offset+self.tile_height/2]
                dist=np.abs(center[0]-normalized_x)+np.abs(center[1]-normalized_y)
                feats[active_y][active_x]=np.exp(-(dist**2)/(2*self.tile_width*self.tile_height))
            else :
                feats[active_y][active_x]=1
        return np.flip(feats,axis=0)