#!/usr/bin/python

import os
import sys
import numpy as np
import caffe
import cv2
import matlab.engine


# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

NET_FILE = os.path.join(os.path.dirname(__file__), "model.prototxt")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.caffemodel")

TILE_SIZE = 256
PADDING_SIZE = 21

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally if BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=16

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

def howes_binarization(img):
	im_mat = matlab.uint8(img[:,:,::-1].tolist())

	# eng declared in main()
	result_mat = eng.binarizeImage(im_mat)
	result_np = np.array(result_mat._data.tolist(), dtype='uint8')
	result_np = result_np.reshape(result_mat.size[1], result_mat.size[0]).transpose()

	# Returns an array of values 0 and 1
	return result_np


def setup_network():
	network = caffe.Net(NET_FILE, WEIGHTS_FILE, caffe.TEST)
	return network


def fprop(network, ims, howes_ims, batchsize=BATCH_SIZE):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]
		sub_howes_ims = ims[idx:idx+batchsize]

		network.blobs["data"].reshape(len(sub_ims), 3, ims[0].shape[1], ims[0].shape[0])
		network.blobs["howes_data"].reshape(len(sub_howes_ims), 3, ims[0].shape[1], ims[0].shape[0])

		for x in range(len(sub_ims)):
			transposed = np.transpose(sub_ims[x], [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed

			transposed_howe = np.transpose(sub_howes_ims[x], [2,0,1])
			transposed_howe = transposed_howe[np.newaxis, :, :, :]
			network.blobs["howes_data"].data[x,:,:,:] = transposed_howe
		idx += batchsize

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs["prob"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * idx / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims, howes_ims):
	all_outputs = fprop(network, ims, howes_ims)
	predictions = np.argmax(all_outputs, axis=1)
	return predictions


def get_subwindows(im):
	height, width, = TILE_SIZE, TILE_SIZE
	y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	bin_ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE) 
			) )
			ims.append(im[y:y+height,x:x+width])
			x += x_stride
		y += y_stride

	for im in ims:
		bin_ims.append(howes_binarization(im))

	return locations, ims, bin_ims


def stich_together(locations, subwindows, size):
	output = np.zeros(size, dtype=np.uint8)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - PADDING_SIZE
		elif y_type == MIDDLE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif y_type == BOTTOM_EDGE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - PADDING_SIZE

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - PADDING_SIZE
		elif x_type == MIDDLE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif x_type == RIGHT_EDGE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - PADDING_SIZE

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output
	

def main(in_image, out_image):
	global eng = matlab.engine.start_matlab()

	image = cv2.imread(in_image, cv2.IMREAD_COLOR)
	image = 0.0039 * (image - 127.)

	network = setup_network()
	locations, subwindows, howes_subwindows = get_subwindows(image)
	binarized_subwindows = predict(network, subwindows, howes_subwindows)
	
	result = stich_together(locations, binarized_subwindows, tuple(image.shape[0:2]))
	result = 255 * result
	cv2.imwrite(out_image, result)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: python hdibco_submission.py in_image out_image [gpu#]"
		print "\tin_image is the input image to be binarized"
		print "\tout_image is where the binarized image will be written to"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If ommitted, CPU mode is used"
		exit(1)
	# only required argument
	in_image = sys.argv[1]

	# attempt to parse an output directory
	out_image = sys.argv[2]

	# use gpu if specified
	try:
		gpu = int(sys.argv[3])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	main(in_image, out_image)
	
