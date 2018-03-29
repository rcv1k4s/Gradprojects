

class Block:
    def __init__(self, dirtyBit,  isValid):
        self.dirtyBit=dirtyBit
        self.isValid = isValid
        self.access=0

    def read(self): 
        self.access = self.access + 1
        
    def isDirty(self):
        return self.dirtyBit
                
    def setValid(self,  valid):
        self.isValid = valid
