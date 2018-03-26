import gzip
k = 0
for line in gzip.open('/home/rchamart/PyCacheSimulator-FT/gcc.gz','rt'):
 data = line.split(' ')
 type = int(data[0], 10)
 addr = int(data[1], 16)
 if (type == 0 or type == 1):
  print (addr)
  k = k + 1
 if k >= 500000:
  break;
