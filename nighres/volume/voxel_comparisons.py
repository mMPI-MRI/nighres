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

def image_pair(contrast_image1, contrast_image2, mask_file=None, comparisons=None, return_data_vectors=True):
	'''
	Directly compare the voxels of one contrast image to another. 
	Intended to be used for comparing two different contrasts in a 
	single subject.
	
	Images MUST be co-registered and of the same space, affine's are not
	checked and no up/down-sampling is performed
	'''

	#TODO: create a dictionary to store output

	d1 = _volume_to_1d(contrast_image1, mask_file=mask_file)
	d2 = _volume_to_1d(contrast_image2, mask_file=mask_file)
	
	d1d2 = np.vstack(d1,d2)
	
	if comparisons is None:
		return d1d2
	else:
		if isinstance(comparisons),basestring):
			comparisons = [comparisons]
		for comparison in comparisons:
			if comparison == "corrcoef":
				corrcoef = np.corrcoef(d1d2)
			#loop over what you wanted to do with your data
			
	
