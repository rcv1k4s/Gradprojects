#!/usr/bin/python3.5

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

import re
from Cache import Cache
import argparse
import gzip
from futaddress import Futaddr 
N = 50000

parser = argparse.ArgumentParser()
parser.add_argument("cache_trace", help="Memory address trace in gzip format")
parser.add_argument("fut_trace", help="Future Memory address trace in gzip format")
parser.add_argument("cache_size", help="Cache size in KB.", type=int)
parser.add_argument("block_size", help="Block size in B.", type=int)
parser.add_argument("set_number", help="set number", type=int)
parser.add_argument("address_bit_size", help="set number", type=int, default=32, action='store', nargs='?')

args = parser.parse_args()

regex = re.compile('[0-9]+')

'''
def readFile(fileAddr):
    instructions=[]
    dataAddr=[]
    with open(fileAddr) as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(' ')
            type = int(data[0],  10)
            addr = int(data[1],  16)
            if(type == 2):
                instructions.append(addr)
            elif(type == 0 or type == 1):
                dataAddr.append(addr)
            else:
                print(data[0])
    return [instructions, dataAddr]
'''

def simulateCaches(file_handler,  i_ch, d_ch):
    
     
    futaddr = Futaddr(N)
    g = 0
    #for line in gzip.open("iaddress",'rt'):
    with open("test/iaddress") as f:
      lines = f.readlines()
      for line in lines:
       tdata = line.split(' ')
       #tftype = int(tdata[0], 10 )
       tfudr =  int(tdata[0], 16)
       futaddr.append(tfudr)
       g = g + 1
       if g>=N:
        break
    #print (len(futaddr.get()))
    #input("here!!!!!!!")
        
    with open("test/iaddress") as trace: #, open("ifaddress-100k") as futtrace:
      lines = trace.readlines()
      for line in lines:
        data = line.split(' ')
        #type = int(data[0], 10)
        addr = int(data[0], 16)
        fdata = line.split(' ')
        #ftype = int(data[0], 10)
        faddr = 0 #int(data[0], 16)
        futaddr.append(faddr)
        farray = [0]*N
        farray = futaddr.get()
        
        #raw_input("Heress!!!") 

        #i_ch.read(addr) # Data read (0) or Data write (1)
        #if (type == 0 or type == 1):
        #print (addr,farray[0:4])
        #input("HERe!!!")
        d_ch.read(addr,farray)

        try:
            miss_rate_d = '{0:.2f}'.format(float(d_ch.miss) * 100 / d_ch.access)
        except ZeroDivisionError:
            miss_rate_d = 'N/A'

        try:
            miss_rate_i = '{0:.2f}'.format(float(i_ch.miss) * 100 / i_ch.access)
        except ZeroDivisionError:
            miss_rate_i = 'N/A'

        #print (d_ch.access)
        #print("{} miss rate : {}, access : {} and {} miss rate : {}, access : {}"
          #    .format(i_ch.name, miss_rate_i, i_ch.access,
           #           d_ch.name, miss_rate_d, d_ch.access), flush=True, end='\r')
    #print()
    #printResult(i_ch)
    #printResult(d_ch)

def printResult(ch):
    print        
    print("-----------------------------")
    print("\tResult for " + ch.name +":")
    print("\tTotal     : " + str(ch.access))
    print("\tMisses     : " + str(ch.miss))
    print("\tHit     : " + str(ch.access - ch.miss))
    print("\tHit Rate : {0:.5}".format(float(ch.access - ch.miss)*100/ch.access))
    print("-----------------------------")


if __name__ == '__main__':

    filePath = args.cache_trace
    futfilepath = args.fut_trace
    cacheSize = args.cache_size * 1024
    blockSize = args.block_size
    setNumber = args.set_number
    address_bit_size = args.address_bit_size

    file_handler = gzip.open(filePath, 'rt')
    fut_file_handler = gzip.open(futfilepath, 'rt')

    l1_ins = Cache(address_bit_size, 'l1_icache',  cacheSize,  blockSize,  setNumber)
    l1_d = Cache(address_bit_size, 'l1_dcache',  cacheSize,  blockSize,  setNumber)
    l1_ins.construct()
    l1_d.construct()
    simulateCaches(file_handler, l1_ins, l1_d)
