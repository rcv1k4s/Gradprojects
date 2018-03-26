'''
MIT License

Copyright (c) 2018 Reza Baharani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 
	A simple cache simulator developed by Reza Baharani
	This code is developed for Computer Architecture subject
	University of North Carolina, Charlotte, USA
'''

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
