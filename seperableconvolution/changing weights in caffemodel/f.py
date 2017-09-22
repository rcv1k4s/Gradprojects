#program to load weights from s.txt as array in to .caffemodel of particular net
import numpy as np
import time
import caffe

f=open('nospace.txt')
lines = f.readlines()
for ln in lines[0:len(lines)]:
	line=ln.strip().split()

b = line
a=np.zeros((285,7,7))
i=0
l=0
while (i<285):
	j=0
	while (j<7):
		k=0
		while (k<7):
			a[i][j][k] = b[l]			
			k=k+1
			l=l+1
		j=j+1
	i=i+1
print a
net = caffe.Net('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet_deploy.prototxt',
              #  '/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet.caffemodel', caffe.TEST)

net.params['conv1'] = co
net.save('myNewTrainModel.caffemodel');

net.save('new_weights.caffemodel')
