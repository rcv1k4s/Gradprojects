#program to load weights from s.txt as array in to .caffemodel of particular net
import numpy as np
import time
import caffe
import os
import argparse
import time

f=open('sepwts.txt')
lines = f.readlines()
for ln in lines[0:len(lines)]:
	line=ln.strip().split()

b = line

print len(b)
a=np.zeros((288,7,7))
i=0
l=0
while (i<288):
	j=0
	while (j<7):
		k=0
		while (k<7):
			a[i][j][k] = float(b[l])			
			k=k+1
			l=l+1
		j=j+1
	i=i+1
#print a
r = np.zeros((96,3,7,7))
i=0
w=0
while (i<96):
	j=0
	while (j<3):
		r[i][j] = a[w]
		w=w+1
		j=j+1
	i=i+1



net = caffe.Net('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet_deploy.prototxt',
                '/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet.caffemodel',caffe.TEST)

b = net.params['conv1'][0].data[...] 


net = caffe.Net('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet_deploy.prototxt',
                '/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet.caffemodel',caffe.TEST)

net.params['conv1'][0].data[...] = r

#print net.params['conv1'][0].data[0]

net.save('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/myNew.caffemodel')

