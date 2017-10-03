import numpy as np
import nibabel as nb
import os
import sys
import cbstools
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

def image_pair(contrast_image1, contrast_image2, mask_file=None, distances=['euclidian','seuclidian','correlation'], return_data_vectors=True, remove_zero_voxels=False):
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
	distance_metrics: str|list
		Single string or list of distances to conduct on input images
		Can be 'all' or one or more of the distance metrics from
		scipy.spatial.distance.cdist:
		'braycurtis', 'canberra', 'chebyshev', 'cityblock',
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
		'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
		'rogerstanimoto', 'russellrao', 'seuclidean',
		'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
		'yule'.
	return_data_vectors: bool (default = True)
		Return 1d data vectors of input image in dictionary as
		"input_vectors"
	remove_zero_voxels: bool (default = False)
		Removes all voxels that equal 0 from BOTH datasets
		Only use if you know that 0 is not an interesting number in
		either of your datasets (i.e., it corresponds to non-data), or
		if you are too lazy to provide a mask and want something quick!
	'''

	import scipy.spatial.distance.cdist as cdist

	v1 = _volume_to_1d(contrast_image1, mask_file=mask_file)
	v2 = _volume_to_1d(contrast_image2, mask_file=mask_file)

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

	if distance_metrics is not None:
		if isinstance(distance_metrics,basestring):
			distance_metrics = [distance_metrics]
		if 'all' in distance_metrics:
			distance_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
			'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
			'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
			'rogerstanimoto', 'russellrao', 'seuclidean',
			'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
			'yule']
		for metric in distance_metrics:
			# res["euclidian_distance"] = np.linalg.norm(v1-v2,2,0)
			res[metric] = cdist(v1,v2,metric=metric)
	return res

def extract_data_multi_image(contrast_image_list, mask_file=None, stack='vertical', start_stop_indices=None):
	'''
	Extract data from multiple image volumes and convert to 1d vector(s). If
	mask_file contains more than one non-zero value, output will be split.
	'''

	if isinstance(contrast_image_list,basestring):
		contrast_image_list = [contrast_image_list]

	mask_img = load_volume(mask_file)
	mask_data = mask_img.get_data()
	mask_ids = np.unique(mask_data)

	for mask_id in mask_ids[mask_ids > 0]:
		seg_mask = np.zeros_like(mask_data)
		seg_mask[mask_data==mask_id] = 1
		vec = _volume_to_1d(contrast_image, mask=seg_mask)
