# Contains Block Cache in class 

class Block:
    def __init__(self, dirtyBit,  isValid):
        self.dirtyBit=dirtyBit
        self.isValid = isValid
        self.access=0

# Beacuse it was not necessary to implement the whole cache hirarchy, for simulating the cache behaviour I used read() instead for type 1 addresses.
    def read(self): 
        self.access = self.access + 1
        
    def isDirty(self):
        return self.dirtyBit
                
    def setValid(self,  valid):
        self.isValid = valid
