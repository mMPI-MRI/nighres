import numpy as np
import pandas as pd
import nibabel as nb
#import os
#import sys
#import cbstools

from ..io import load_volume, save_volume
#from ..utils import _output_dir_4saving

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

def extract_data_multi_image(contrast_image_list, mask_file=None, image_thr=None):
	'''
	Extract data from multiple image volumes and convert to 1d vector(s). If
	mask_file contains more than one non-zero value, output will be grouped
	in columns according to sorted (increasing) mask_ids. Column indices provided
	in output as mask_id_start_stop

	mask_file: niimg
		Mask {0,1,...,n} of the same dimensions as input contrast_image1/2
		Multiple indices >0 are valid. Start/stop positions in 2nd dimension
		of output data_matrix.

	image_thr: int
		This value is used to construct a mask from the first image in
		contrast_image_list (anything BELOW this value is set to 0, i.e., masked out)
		ONLY used when mask_file is not provided. Don't be lazy, provide
		a mask if you can.
	'''

	if isinstance(contrast_image_list,basestring):
		contrast_image_list = [contrast_image_list]

	if mask_file is not None:
		mask_img = load_volume(mask_file)
		mask_data = mask_img.get_data()
	else:
		mask_data = np.ones(load_volume(contrast_image_list[0]).shape)
		if image_thr is not None:
			mask_data[load_volume(contrast_image_list[0]).get_data() < image_thr] = 0

	mask_ids = np.unique(mask_data)
	mask_ids.sort()

	# pre=allocate an array for data
	# rows are contrast images, cols are the data from each of the segs
	# fastest way to do this, but obviously not the most straight forward
	# to work with later - this is damn fast though
	# TODO: make easier to use/interpret (lists of arrays?)
	# # mod: create an empty list of lists and then fill with arrays of known size
	# # mod: data_matrix_list = [[] for _ in range(np.sum(mask_ids>0))]

	data_matrix = np.zeros((len(contrast_image_list),np.sum(mask_data.flatten()>0)))*np.nan
	mask_id_start_stop  = np.zeros((np.sum(mask_ids>0),3))*np.nan

	for image_idx, contrast_image in enumerate(contrast_image_list):
		start = 0

		for mask_id_idx, mask_id in enumerate(mask_ids[mask_ids > 0]):
			#print("mask id: {}".format(mask_id))
			seg_mask = np.zeros_like(mask_data)
			seg_mask[mask_data==mask_id] = 1
			seg_vec = _volume_to_1d(contrast_image, mask=seg_mask)
			#print(seg_vec.shape)
			stop = len(seg_vec)+start
			#print("{} - {}".format(start,stop))
			data_matrix[image_idx,start:stop] = seg_vec

			# construct a lookup for the mask ids and their respective start
			# and stop columns, but only on the first image since they will
			# all be the same
			if image_idx == 0:
				mask_id_start_stop[mask_id_idx] = np.array([mask_id,start,stop])
			start = np.copy(stop)
	return data_matrix, mask_id_start_stop.astype(int)

def element_lm(data_matrix_full,descriptives,formula,output_vars,contrast_images_colname_head='contrast_image_'):
	'''
	Element-wise OLS linear model using statsmodels.formula.api.lm . Will correctly treat
	from 1-3 dimensions if 0th dim is always contrast_images, 1st dim is always elements,
	2nd always subjects. Works equally well with vectors of data, data from image
	volumes, and data from vertices.

	data_matrix_full: np.ndarray
			Full matrix of data from each subject and contrast provided.
			0th dim: contrast images
			1st dim: data elements (voxels)
			2nd dim: subjects (or timepoints, or both)
	descriptives: csv|pd.DataFrame
		.csv file or pandas dataframe containing:
			1) full path(s) to subject contrast images (1 or more)
			2) additional demographic, group, or control variables for analyses
	formula: str
		Written linear model of the form 'Y ~ X + Z'
	output_vars: str|list
		Variables that are of interest for output maps (t/p)
		Intercept will automatically be included in the output so do not add it here
	'''
	#TODO: make multiprocessing friendly! (likely with change to create an interim storage type for big data? .hdf5?)

	import statsmodels.formula.api as smf
	from __future__ import print_function
	from sys import stdout as stdout
	#from multiprocessing import Pool

	if np.ndim(data_matrix_full) == 1:
		data_matrix_full = data_matrix_full[:,np.newaxis,np.newaxis]
	elif np.ndim(data_matrix_full) == 2:
		data_matrix_full = data_matrix_full[:,:,np.newaxis] #this may not work for all cases, depending on the assumptions made

	if isinstance(output_vars,basestring):
		output_vars = [output_vars]
	output_vars.append('Intercept') #add the intercept for output too

	if isinstance(descriptives,basestring): #should be a csv file, load it
		df = pd.read_csv(descriptives,header=0)
	elif isinstance(descriptives, pd.DataFrame):
		df = descriptives.copy()

	# TODO: check stuffs
	# check to make sure that the dimensions are compatible between descriptives
	# and data_matrix_full

	res_p = np.zeros((len(output_vars),data_matrix_full.shape[1]))*np.nan #each output var gets its own row
	res_t = np.copy(res_p)*np.nan
	res_rsquared_adj = np.zeros((data_matrix_full.shape[1]))*np.nan #single row of R2, only one per model
	#print(res_p.shape)

	# this is likely quite slow, since we run linear models separatenly for each element :-/
	for el_idx in range(data_matrix_full.shape[1]):
		vdata = np.transpose(np.squeeze(data_matrix_full[:,el_idx,:]))
		df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]] = vdata #put the data where the contrast_images were
		lmf = smf.ols(formula=formula,data=df).fit()
		for output_var_idx, output_var in enumerate(output_vars):
			#return df
			#print df['contrast_image_1'][0] + df['contrast_image_2'][0]
			#res_p[output_var_idx,el_idx] = df['contrast_image_1'][0] + df['contrast_image_2'][0]
			res_p[output_var_idx,el_idx] = lmf.pvalues[output_var]
			res_t[output_var_idx,el_idx] = lmf.tvalues[output_var]
		res_rsquared_adj[el_idx] = lmf.rsquared_adj
#		progress = (el_idx + 1) / len(data_matrix_full.shape[1])
		print(" Processed: {0}/{1}".format(el_idx,data_matrix_full.shape[1]),end='\r')
        #stdout.write(" Processed: {0}/{1} {2}".format(el_idx,data_matrix_full.shape[1],"\r"))
        stdout.flush()

	res = {}
	res['tvalues'] = res_t
	res['pvalues'] = res_p
	res['rsquared_adj'] = res_rsquared_adj
	res['variable_names'] = output_vars
	return res

def extract_data_group(descriptives,contrast_images_colname_head='contrast_image_',mask_file=None):
	'''
	Extract data from multiple subjects with one or more different contrast
	images. Automatically recognises and pulls multiple contrast images if
	present, with column names starting with contrast_images_colname_head.

	descriptives: csv|pd.DataFrame
		.csv file or pandas dataframe containing:
			1) full path(s) to subject contrast images (1 or more)
			2) additional demographic, group, or control variables for analyses
	contrast_images_colname_head: str
		Unique text for name(s) of column(s) where subject contrast images are
		listed. e.g. contrast_image_ for: ['contrast_image_FA',contrast_image_T1']

	Returns: data_matrix_full (3d np.ndarray), mask_id_start_stop (2d np.ndarray)
		Full matrix of data from each subject and contrast provided.
		0th dim: contrast images
		1st dim: data elements (voxels)
		2nd dim: subjects (or timepoints, or both)
	'''

	if isinstance(descriptives,basestring): #should be a csv file, load it
		df = pd.read_csv(descriptives,header=0)
	elif isinstance(descriptives, pd.DataFrame): #check if this is a pd.DataFrame
		df = descriptives

	df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
	contrasts_list = df_contrasts_list.values.tolist()

	for contrasts_idx,contrasts in enumerate(contrasts_list): #TODO: this will fail with only a single individual's data
		data_matrix, mask_id_start_stop = extract_data_multi_image(contrasts,mask_file=mask_file,image_thr=None) #no image thresh in group data, all data needs to be the same so use a mask
		#if this is the first time through, we use the shape of the data_matrix to set our output size
		if contrasts_idx == 0:
			data_matrix_full = np.zeros((data_matrix.shape + (np.shape(contrasts_list)[0],)))
		data_matrix_full[:,:,contrasts_idx] = data_matrix

	return data_matrix_full, mask_id_start_stop

def write_element_results(res,descriptives,output_dir,file_name_head,contrast_images_colname_head='contrast_image_',mask_file=None):
	'''
	Write statistical results (pvals,tvals,rsquared_adj) back to the type of file
	from which they were generated.
	'''
	import os

	if isinstance(descriptives,basestring): #should be a csv file, load it
		df = pd.read_csv(descriptives,header=0)
	elif isinstance(descriptives, pd.DataFrame):
		df = descriptives

	#determine type of file
	df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
	contrasts_list = df_contrasts_list.values.tolist()
	#return df_contrasts_list
	if isinstance(contrasts_list, basestring):
		fname = contrasts_list[0]
	else:
		fname = contrasts_list[0][0]
	ext = os.path.basename(fname).split('.',1)

	img = load_volume(fname)
	head = img.get_header()
	aff = img.get_affine()
	out_data = np.zeros(img.shape)
	if ext is 'nii.gz' or 'nii':
		if mask_file is not None:
			mask = load_volume(mask_file).get_data().astype(bool)
		else:
			mask = np.ones(out_data.shape).astype(bool)

		for var_idx, variable in enumerate(res['variable_names']):
			#write the volume for pvals
			out_data[mask] = res['pvalues'][var_idx]
			out_fname = os.path.join(output_dir,file_name_head + '_' + variable + '_p.nii.gz')
			head['cal_max'] = out_data.max()
			head['cal_min'] = out_data.min()
			img = nb.Nifti1Image(out_data,aff,header=head)
			save_volume(out_fname,img)
			print(out_fname)

			#write the volume for tvals
			out_data[mask] = res['tvalues'][var_idx]
			out_fname = os.path.join(output_dir,file_name_head + '_' + variable + '_t.nii.gz')
			head['cal_max'] = out_data.max()
			head['cal_min'] = out_data.min()
			img = nb.Nifti1Image(out_data,aff,header=head)
			save_volume(out_fname,img)
			print(out_fname)

		#write the r2 volume
		out_data[mask] = res['rsquared_adj']
		out_fname = os.path.join(output_dir,file_name_head + '_' + 'model' + '_r2adj.nii.gz')
		head['cal_max'] = out_data.max()
		head['cal_min'] = out_data.min()
		img = nb.Nifti1Image(out_data,aff,header=head)
		save_volume(out_fname,img)
		print(out_fname)
	elif ext is 'txt': #working with vertex files
		pass


def compute_image_pair_stats(data_matrix, example_data_file, mask_id_start_stop=None, mask=None):
	'''
	data_matrix: np.ndarray() (3d)
		A matrix of dimensions contrast (2) by voxel/element (n) by subject
	mask_id_start_stop: np.ndarray() (2d)
		Matrix of mask_id values along with the start and stop positions along the
		data_matrix for use if there is more than one index in the mask (i.e., a
		segmentation)
		If none, assumes all data are from a single mask
	mask:
	'''

	# to bring data back into the volume space, again assuming full co-reg
	# needs to be looped over sorted indices again as necessary
	# stat_map = np.zeros_like(contrast_image_list[0])
	# stat_map[mask] = stats

	# initialise the output stat map
	img = load_volume(example_data_file)
	stat_map = np.zeros(img.shape)
	aff = img.get_affine()
	head = img.header
	del img

	if mask is not none:
		mask_img = load_volume(mask)
		mask_data = mask_img.get_data()
		del mask_img
		mask_ids = np.unique(mask_data)
		mask_ids.sort()
	else:
		mask_ids = [[1]]
		mask_ids = np.ones_like(stat_map).astype(bool)

	for mask_id_idx, mask_id in enumerate(mask_ids[mask_ids > 0]):
		#do stuff here, only because you can output summary stats, otherwise why are you breaking this up into different segments... since it is all voxel-wise anyways
		# TODO: reconsider
		pass
	return None
