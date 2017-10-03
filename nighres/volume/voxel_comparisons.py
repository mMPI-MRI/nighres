import numpy as np
import nibabel as nb
import os
import sys
#import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving

def _volume_to_1d(volume_file, mask=None):
	'''
	Converts 3D volume to 1d vector
	'''

	img = load_volume(volume_file)
	data = img.get_data()

	# if there is a mask, convert it to boolean and select only that data
	if mask is not None:
		if isinstance(mask,basestring):
			mask_img = load_volume(mask)
			mask_bool = mask_img.get_data().astype(bool)
			return data[mask_bool]
		elif isinstance(mask,np.ndarray):
			mask_bool = mask.astype(bool)
			return data[mask_bool]
	else:
		return data.flatten()

def image_pair(contrast_image1, contrast_image2, mask_file=None, return_data_vectors=True, remove_zero_voxels=False):
	'''
	Directly compare the voxels of one contrast image to another.
	Intended to be used for comparing two different contrasts in a
	single subject. Returns correlation and data vectors as requested

	Images MUST be co-registered and of the same space, affine's are not
	checked and no up/down-sampling is performed

	contrast_image1: niimg
		First input image
	contrast_image2: niimg
		Second input image
	mask_file: niimg
		Mask {0,1} of the same dimensions as input contrast_image1/2
	return_data_vectors: bool (default = True)
		Return 1d data vectors of input image in dictionary as
		"input_vectors"
	remove_zero_voxels: bool (default = False)
		Removes all voxels that equal 0 from BOTH datasets
		Only use if you know that 0 is not an interesting number in
		either of your datasets (i.e., it corresponds to non-data), or
		if you are too lazy to provide a mask and want something quick!
	'''
	v1 = _volume_to_1d(contrast_image1, mask=mask_file)
	v2 = _volume_to_1d(contrast_image2, mask=mask_file)

	# if we are sure that 0 is meaningless and not to be compared,
	# remove from both vectors
	if remove_zero_voxels:
		v2[v1==0] = np.nan
		v2[v2==0] = np.nan
		v1[v1==0] = np.nan
		v1[v2==0] = np.nan
		v1 = v1[~np.isnan(v1)]
		v2 = v2[~np.isnan(v2)]

	res = {}
	res['contrast_image1'] = contrast_image1
	res['contrast_image2'] = contrast_image2
	res['zeros_removed'] = remove_zero_voxels
	res["corrcoef"] = np.corrcoef(v1,v2) #compute the correlation coef

	if return_data_vectors:
		res["input_vectors"] = np.vstack([v1,v2])

	return res

def extract_data_multi_image(contrast_image_list, mask_file=None, image_zero_value=None):
	'''
	Extract data from multiple image volumes and convert to 1d vector(s). If
	mask_file contains more than one non-zero value, output will be grouped
	in columns according to sorted (increasing) mask_ids. Column indices provided
	in output as mask_id_start_stop

	image_zero_value: int
		This value is used to construct a mask from the first image in
		contrast_image_list (this value is set to 0, i.e., masked out)
		ONLY used when mask_file is not provided. Don't be lazy, provide
		a mask if you can.
	'''

	if isinstance(contrast_image_list,basestring):
		contrast_image_list = [contrast_image_list]

	if mask_file is not None:
		mask_img = load_volume(mask_file)
		mask_data = mask_img.get_data()
	else:
		mask_data = np.ones_like(load_volume(contrast_image_list[0]).shape)
		if image_zero_value is not None:
			mask_data[load_volume(contrast_image_list[0]).get_data() == image_zero_value] = 0

	mask_ids = np.unique(mask_data)
	mask_ids.sort()

	# pre=allocate an array for data
	# rows are contrast images, cols are the data from each of the segs
	# fastest way to do this, but obviously not the most straight forward
	# to work with later - this is damn fast though
	# TODO: make easier to use/interpret (lists of arrays)
	# # mod: create an empty list of lists and then fill with arrays of known size
	# # mod: res_list = [[] for _ in range(np.sum(mask_ids>0))]

	res = np.zeros((len(contrast_image_list),np.sum(mask_data.flatten()>0)))*np.nan
	mask_id_start_stop  = np.zeros((np.sum(mask_ids>0),3))*np.nan

	for image_idx, contrast_image in enumerate(contrast_image_list):
		start = 0

		for mask_id_idx, mask_id in enumerate(mask_ids[mask_ids > 0]):
			print("mask id: {}".format(mask_id))
			seg_mask = np.zeros_like(mask_data)
			seg_mask[mask_data==mask_id] = 1
			seg_vec = _volume_to_1d(contrast_image, mask=seg_mask)
			print(seg_vec.shape)
			stop = len(seg_vec)+start
			print("{} - {}".format(start,stop))
			res[image_idx,start:stop] = seg_vec

			# construct a lookup for the mask ids and their respective start
			# and stop columns, but only on the first image since they will
			# all be the same
			if image_idx == 0:
				mask_id_start_stop[mask_id_idx] = np.array([mask_id,start,stop])
			start = np.copy(stop)
	return res, mask_id_start_stop.astype(int)
