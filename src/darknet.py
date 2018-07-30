from __future__import division
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

def parse_cfg(cfgfile):

	file = open(cfgfile,'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if len(x) > 0]
	lines = [x for x in lines if x[0] != '#']
	lines = [x.rstrip().lstrip() for x in lines]

	block = {}

	blocks = []

	for line in lines:
		if line[0] == "[":
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block["type"] = line[1:-1].rstrip()
		else:
			key,value = line.split("=")
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks


def create_modules(blocks):

	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3
	output_filters = []

    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        if (x["type"] == "convolutional"):

        	#Get the info about the layer
        	activation = x["activation"]
        	try:
        		batch_normalize = int(x["batch_normalize"])
        		bias =False
        	except:
        		batch_normalize = 0
        		bias = True

        	filters = int(x["filters"])
        	padding = int(x["pad"])
        	kernel_size = int(x["size"])
        	stride = int(x["stride"])

        	if padding:
        		pad = (kernel_size -1)//2
        	else:
        		pad = 0

        	#Add the convolutional layer
        	conv = nn.Conv2d(prev_filters , filters , kernel_size , stride , pad , bias = bias)
        	module.add_module("batch_norm_{0}".format(index) , bn)

        	if batch_normalize:

        		bn = nn.BatchNorm2d(filters)
        		module.add_module("batch_norm_{0}".format(inedx),bn)

        	#check the activation
        	#it is either Linear or a leaky ReLU for YOLO
			if activation == "leaky":

				activn = nn.LeakyReLU(0.1 , inplace =True)
				module.add_module("leaky_{0}".format(index),activn)

			#If it is an upsampling layer
			#Use Bilinear2DUpsampling
		elif (X["type"] == "upsample"):

			stride = int(x["stride"])
			upsample = nn.Upsample(scale_factor = 2 , mode = 'bilinear')
			module.add_module("upsample_{}".format(index), upsample)