#!/usr/bin/python3.5
'''Data Cache Simulator
   Farthest in Future Replacement policy
   Ramachandra Vikas Chamarthi
   vikaschamarthi240@gmail.com
   Graduate Research Assistant 
   The UNC Charlotte'''
   
import re
from Cache import Cache
import argparse
import gzip
from futaddress import Futaddr 

N = 50000 #Number of Addresses to look up in future

parser = argparse.ArgumentParser()
parser.add_argument("cache_trace", help="Memory address trace in gzip format")
parser.add_argument("cache_size", help="Cache size in KB.", type=int)
parser.add_argument("block_size", help="Block size in B.", type=int)
parser.add_argument("set_number", help="set number", type=int)
parser.add_argument("address_bit_size", help="set number", type=int, default=32, action='store', nargs='?')

args = parser.parse_args()

regex = re.compile('[0-9]+')


def simulateCaches(cf, i_ch, d_ch):
    
    N 
    futaddr = Futaddr(N)
    g = 0
    print (cf)
    with open(cf) as f:
      lines = f.readlines()
      for line in lines:
       tdata = line.split(' ')
       tfudr =  int(tdata[0], 16)
       futaddr.append(tfudr)
       g = g + 1
       if g>=N:
        break
    
        
    with open(cf) as trace: 
      lines = trace.readlines()
      for line in lines:
        data = line.split(' ')
        
        addr = int(data[0], 16)
        fdata = line.split(' ')
        
        faddr = 0 
        futaddr.append(faddr)
        farray = [0]*N
        farray = futaddr.get()
        
        
        d_ch.read(addr,farray)

        try:
            miss_rate_d = '{0:.2f}'.format(float(d_ch.miss) * 100 / d_ch.access)
        except ZeroDivisionError:
            miss_rate_d = 'N/A'

            
        print("{} miss rate : {}, access : {}"
              .format(d_ch.name, miss_rate_d, d_ch.access), flush=True, end='\r')
    print()
    printResult(i_ch)
    printResult(d_ch)

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
    cacheSize = args.cache_size * 1024
    blockSize = args.block_size
    setNumber = args.set_number
    address_bit_size = args.address_bit_size

    print (filePath)
    input("!!!!!!!!!!!!!")
    l1_ins = Cache(address_bit_size, 'l1_icache',  cacheSize,  blockSize,  setNumber)
    l1_d = Cache(address_bit_size, 'l1_dcache',  cacheSize,  blockSize,  setNumber)
    l1_ins.construct()
    l1_d.construct()
    simulateCaches( filePath, l1_ins, l1_d)
