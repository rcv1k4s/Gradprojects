#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import time
import caffe


caffe.set_mode_gpu();
caffe.set_device(0);

#load the model
net = caffe.Net('/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet_deploy.prototxt',
                '/home/nvidia/caffe/models/SqueezeNet-Deep-Compression/SqueezeNet.caffemodel',
                caffe.TEST)
print net.params['conv1'][0].data
# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/nvidia/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('/home/nvidia/caffe/examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)
start = time.time()
#compute
out = net.forward()

# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print out['pool_final'].argmax()

#print predicted labels
labels = np.loadtxt("/home/nvidia/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['pool_final'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

print("Done in %.2f s." % (time.time() - start))

