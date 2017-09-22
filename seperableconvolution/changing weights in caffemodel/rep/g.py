import numpy as np
import time
import caffe
import os
import argparse
import time
import sys
import json

net = caffe.Net('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet_deploy.prototxt',
                '/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet.caffemodel',caffe.TEST)


a = net.params['conv1'][0].data[...] 
i=0

while i<96:
	j=0
	while j<3:
		print a[i][j][:][:]
		j=j+1
	i=i+1
