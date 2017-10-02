import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving

def _volume_to_1d(volume_file, mask_file=None):
	'''
	Converts 3D volume to 1d vector
	'''
	
	img = load_volume(volume_file)
	data = img.get_data()
	
	# if there is a mask, convert it to boolean and select only that data
	if mask is not None:
		mask = load_volume(mask_file)
		mask_bool = img.get_data().astype(bool)
		return data[mask_bool]
	else:
		return data.flatten()		

def image_pair(contrast_image1, contrast_image2, mask_file=None, comparisons=['corrcoef','euclidian_distance'], return_data_vectors=True, remove_zero_voxels=False):
	'''
	Directly compare the voxels of one contrast image to another. 
	Intended to be used for comparing two different contrasts in a 
	single subject.
	
	Images MUST be co-registered and of the same space, affine's are not
	checked and no up/down-sampling is performed
	
	contrast_image1: niimg
		First input image
	contrast_image2: niimg
		Second input image
	mask_file: niimg
		Mask {0,1} of the same dimensions as input contrast_image1/2
	comparisons: str|list
		Single string or list of comparisons to conduct on input images
		Can be one or more of ['corrcoef','euclidian_distance']
	return_data_vectors: bool (default = True)
		Return 1d data vectors of input image in dictionary as 
		"input_vectors"
	remove_zero_voxels: bool (default = False)
		Removes all voxels that equal 0 from BOTH datasets
		Only use if you know that 0 is not an interesting number in 
		either of your datasets (i.e., it corresponds to non-data), or 
		if you are too lazy to provide a mask and want something quick!
	'''

	d1 = _volume_to_1d(contrast_image1, mask_file=mask_file)
	d2 = _volume_to_1d(contrast_image2, mask_file=mask_file)
	
	if remove_zero_voxels:
		d2[d1==0] = np.nan
		d2[d2==0] = np.nan
		d1[d1==0] = np.nan
		d1[d2==0] = np.nan
		d1 = d1[~np.isnan(d1)]
		d2 = d2[~np.isnan(d2)]
	
	res = {}
	res['contrast_image1'] = contrast_image1
	res['contrast_image2'] = contrast_image2
	res['zeros_removed'] = remove_zero_voxels
	
	if return_data_vectors:
		res["input_vectors"] = np.vstack([d1,d2])
	
	if comparisons is None:
		return res
	else:
		if isinstance(comparisons),basestring):
			comparisons = [comparisons]
		for comparison in comparisons:
			if comparison == "corrcoef":
				res["corrcoef"] = np.corrcoef(d1,d2)
			if comparison = "euclidian_distance":
				res["euclidian_distance"] = np.linalg.norm(d1-d2,2,0)
	return res
