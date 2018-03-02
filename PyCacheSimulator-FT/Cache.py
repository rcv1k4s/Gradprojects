""" Contains Class construct of cache and functions to extract tag and Index values """

from Block import Block
import  math

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
        self.access = 0;
        self.miss = 0;
        
    def construct(self):
        for i in range(0,  (self.index)):
            for j in range(0,  self.associativity):
                b = Block(0,  0)
                self.sets[i,  j]={0:b} #tag value, block
        self.calculateTagMask();
        self.calculateIndexMask();
                
    def calculateTagMask(self):
        for i in range(0,  self.tagBitSize):
            self.tagMask = self.tagMask << 1;
            self.tagMask = self.tagMask | 1;
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
        h = [0] * 4
        whichBlockIsBetter = -1 # associativity number
        FT_Set = -1
        val = -1
        leastLRUVal = -1
        b = Block(0,  1)
        for i in range(0,  self.associativity):
            if(whichBlockIsBetter == -1):
                data = self.sets[index,  i]
                block = list(data.values())[0]
                if (block.isValid == True):
                    if(leastLRUVal < 0):
                     k = 0
                     farthest = 0
                     for addr in faddr:
                      indexnew = (self.extractIndexValue(addr)>>self.offsetBitSize)
                      tagnew = self.extractTagValue(addr)
                      if tag==tagnew and index==indexnew:
                       farthest=max(farthest, k)
                      k = k+1
                     h[i] = farthest
                else:
                    whichBlockIsBetter = i
                    break
                    
        if(whichBlockIsBetter == -1):
            FT_Set = h.index(max(h))
            self.sets[index,  FT_Set] = {tag : b}
            val = FT_Set + 1
        else:
            self.sets[index,  whichBlockIsBetter] = {tag : b}
            val = whichBlockIsBetter + 1
            
        return val
      

    def read(self,  address, futaddr):
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
                        print (0)
                else:
                    if(i == (self.associativity-1)):
                        print (self.replaceBlock(tag, index,futaddr))
        if(not(hit)):
            self.miss = self.miss + 1
