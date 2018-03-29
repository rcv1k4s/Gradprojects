
from Block import Block
import  math
from futaddress import Futaddr 
import numpy as np

class Cache:
    def __init__(self,  addressBitSize, name,  size,  block_size,  associativity):
        self.name = name
        self.addressBitSize = addressBitSize
        self.size = size
        self.block_size=block_size
        self.associativity = associativity
        self.index = int(size /(block_size * associativity))
        self.indexBitSize = int(math.log(self.index,  2))
        self.offsetBitSize = int(math.log(block_size,  2))
        self.tagBitSize = self.addressBitSize-(self.offsetBitSize + self.indexBitSize)
        self.sets={}
        self.tagMask=0
        self.indexMask=0
        self.access = 0
        self.miss = 0
        
    def construct(self):
        for i in range(0,  (self.index)):
            for j in range(0,  self.associativity):
                b = Block(0,  0)
                self.sets[i,  j]={0:b} #tag value, block
        self.calculateTagMask()
        self.calculateIndexMask()
                
    def calculateTagMask(self):
        for i in range(0,  self.tagBitSize):
            self.tagMask = self.tagMask << 1
            self.tagMask = self.tagMask | 1
        for i in range(0,  self.addressBitSize - self.tagBitSize):
            self.tagMask = self.tagMask << 1
            
    def extractTagValue(self,  addr):
        return (addr & self.tagMask)
        
    def extractIndexValue(self,  addr):
        return (addr & self.indexMask)
    
    def calculateIndexMask(self):
        for i in range(0,  self.indexBitSize):
            self.indexMask = self.indexMask << 1
            self.indexMask = self.indexMask | 1
        for i in range(0,  self.offsetBitSize):
            self.indexMask = self.indexMask << 1
            
    def replaceBlock(self,  tag,  index, faddr):
        h = [-1] * 4
        tagas = [0] * 4
        oldtags = [0] * 4
        whichBlockIsBetter = -1 # associativity number
        FT_Set = -1
        val = -1
        leastLRUVal = -1
        b = Block(0,  1)
        da = [0] * self.associativity
        ad = [0] * self.associativity
        for i in range(0,  self.associativity):
         d = self.sets[index, i]
         dai = list(d.keys())[0]
         da[i] = dai
         adblock = list(d.values())[0]
         ad[i] = adblock.access
        zas = np.array([ad[0],ad[1],ad[2],ad[3]])
        yz = zas.argsort()
        xz = yz.argsort()

        for i in range(0,  self.associativity):
            if(whichBlockIsBetter == -1):
                data = self.sets[index,  i]
                tg = list(data.keys())[0]
                block = list(data.values())[0]
                if (block.isValid == True):
                    if(leastLRUVal < 0):
                     k = 0
                     farthest = 0
                     for addr in faddr:
                      indexnew = (self.extractIndexValue(addr)>>self.offsetBitSize)
                      tagnew = self.extractTagValue(addr)
                      if tg==tagnew and index==indexnew:
                       farthest=max(farthest, k)
                      k = k+1
                     if farthest ==0:
                      farthest = 110000 - block.access
                     h[i] = farthest
                     tagas[i] = tg

                     #farthest= farthest + 1
                else:
                    whichBlockIsBetter = i
                    break
                    
        if(whichBlockIsBetter != -1):
         #print(-1)
         self.sets[index,  whichBlockIsBetter] = {tag : b}
         val = whichBlockIsBetter + 1
        else:
         FT_Set = h.index(max(h))
         self.sets[index,  FT_Set] = {tag : b}
         val = FT_Set + 1
          
        return val,da,xz,h
      

    def read(self,  address, futaddr, print_res = True):
        index = (self.extractIndexValue(address)>>self.offsetBitSize)
        tag = self.extractTagValue(address)
        self.access = self.access + 1
        hit = False;
        for i in range(0,  self.associativity):
            data = self.sets[index,  i]
            block = data.get(tag);
            if(hit == False):
                if(block):
                    if (block.isValid == True):
                        block.read();
                        hit = True
                        #print (0)
                else:
                    if(i == (self.associativity-1)):
                        o,t,k,e = (self.replaceBlock(tag, index,futaddr))
                        #print(index,tag,k[0],k[1],k[2],k[3],t[0],t[1],t[2],t[3],o)
        if(not(hit)):
            self.miss = self.miss + 1
