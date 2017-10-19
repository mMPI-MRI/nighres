from __future__ import print_function
import numpy as np
import pandas as pd
import nibabel as nb
import os

#import os
#import sys
#import cbstools

from ..io import load_volume, save_volume
#from ..utils import _output_dir_4saving

def generate_group_mask(contrast_images,thresholds=None,contrast_images_colname_head='img_',zero_below=True):
	'''
	Generate a binary group mask for the contrast images provided in the dataframe.
	Thresholds are applied to contrast_images in the same order that they exist
	in the dataframe. Produces a binary mask where all conditions are met in all
	input images.
	'''

	if isinstance(contrast_images,pd.DataFrame):
		df = contrast_images
	else:
		print('Non-dataframe input is not currently supported')
		return None
	#elif isinstance(contrast_images,list):
	#    df = pd.DataFrame(contrast_images,columns=['img_unknown'])
	#    pass

	df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
	contrasts_list = df_contrasts_list.values.tolist()
	num_cons = df_contrasts_list.shape[1]

	if len(thresholds) != num_cons:
		print('You have not supplied enough threshold values {} for the number of contrasts included in your file {}'.format(len(thresholds),num_cons))
		return None

	for contrasts_idx,contrasts in enumerate(contrasts_list):
		if contrasts_idx == 0:
			mask_data = threshold_image_list(contrasts,thresholds=thresholds,zero_below=zero_below,verbose=True)
		else:
			mask_data = np.multiply(mask_data,threshold_image_list(contrasts,thresholds=thresholds,zero_below=zero_below,verbose=True))
	img = load_volume(contrasts[0])
	head = img.get_header()
	aff = img.get_affine()
	img = nb.Nifti1Image(mask_data,aff,header=head)
	return img

#TODO: make internal
def threshold_image_list(img_list,thresholds=None,zero_below=True,verbose=True):
	# create a new mask based on the thresholds that were passed
	binarise = True
	if thresholds is None: #if nothing supplied, use 0
		thresholds = np.zeros(len(img_list))
	print(thresholds)
	if verbose:
		print('Generating combined binary image mask based on supplied threshold(s)')
		print('  Files: \n    {0}'.format(img_list))
		print('  Image threshold values: {0}'.format(thresholds))
	for thr_idx,thresh_val in enumerate(thresholds):
		if thr_idx == 0:
			mask_data = threshold_image(img_list[thr_idx],threshold=thresholds[thr_idx],binarise=True,zero_below=zero_below)
		else:
			mask_data = np.multiply(mask_data,threshold_image(img_list[thr_idx],threshold=thresholds[thr_idx],binarise=True,zero_below=zero_below))
	return mask_data

def threshold_image(img,threshold=0,binarise=True,zero_below=True):
    '''
    Set image to 0 below or above a threshold. Binarise the output by default.

    img: str|np.ndarray
        Path to image file or np.ndarray to threshold
    threshold: int|float
        Cut-off threshold value
    binarise: bool
        Convert all values outside of the threshold to 1
    zero_below: bool
        Set values below threshold to 0. If false, set values above threshold
        to 0
    returns: np.ndarray
        Thresholded array (binary or not)
    '''

    if isinstance(img,basestring):
        mask_img = load_volume(img)
        data = mask_img.get_data()
    if isinstance(img,np.ndarray):
        data = img
    if zero_below:
        data[data<threshold] = 0
    if not zero_below:
        data[data>threshold] = 0
    if binarise:
        data = data.astype(bool).astype(int)
    return data

def _mm2vox(aff,pts):
    import nibabel as nb
    import numpy as np
    #convert xyz coords from mm to voxel space coords
    return (nb.affines.apply_affine(np.linalg.inv(aff),pts)).astype(int)
def _vox2mm(aff,pts):
    import nibabel as nb
    #convert from voxel coords space back to mm space xyz
    return nb.affines.apply_affine(aff,pts)

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

def image_pair(contrast_image1, contrast_image2, mask_file=None,
               return_data_vectors=True, remove_zero_voxels=False):
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

	res = {'contrast_image1':contrast_image1, 'contrast_image2':contrast_image2, 'zeros_removed':remove_zero_voxels, 'corrcoef':np.corrcoef(v1,v2)}
	if return_data_vectors:
		res["input_vectors"] = np.vstack([v1,v2])
	return res

def extract_data_multi_image(contrast_image_list, mask_file=None, image_thresholds=None, zero_below=True,verbose=False):
	'''
	Extract data from multiple image volumes and convert to 1d vector(s). If
	mask_file contains more than one non-zero value, output will be grouped
	in columns according to sorted (increasing) mask_ids. Column indices provided
	in output as mask_id_start_stop

	mask_file: niimg
		Mask {0,1,...,n} of the same dimensions as input contrast_image1/2
		Multiple indices >0 are valid. Start/stop positions in 2nd dimension
		of output data_matrix.

	image_thresholds: np.array.astype(int|float)
		These values are used to construct masks from the images in
		contrast_image_list. The final mask is the product of all generated
		masks.
		ONLY used when mask_file is not provided. Don't be lazy, provide
		a mask if you can.
    zero_below: bool
        If mask_threshold ~= None, zero below the threshold (or above if False)

    Returns: dict
        data_matrix
        	['data_matrix_full'] - data matrix for this individual
        	['contrast_names'] - filenames
        	['mask_id_start_stop'] - label idx, start, and stop indices in data_matrix_full for non-binary segmentation
	'''
	if isinstance(contrast_image_list,basestring):
		contrast_image_list = [contrast_image_list]

	if mask_file is not None:
		mask_img = load_volume(mask_file)
		mask_data = mask_img.get_data()
	else:
		mask_data = np.ones(load_volume(contrast_image_list[0]).shape)
		if image_thresholds is not None:

			#convert to list if single value passed
			if isinstance(image_thresholds,int) or isinstance(image_thresholds,float):
				image_thresholds = [image_thresholds]
			elif len(image_thresholds) == 1:
 				image_thresholds = [image_thresholds]

			# create a new mask based on the thresholds that were passed
			if verbose:
				print('Generating combined binary image mask based on supplied threshold(s)')
				print('  Files: \n    {0}'.format(contrast_image_list))
				print('  Image threshold values: {0}'.format(image_thresholds))
			for thr_idx,thresh_val in enumerate(image_thresholds):
				if thr_idx == 0:
					mask_data = threshold_image(contrast_image_list[thr_idx],threshold=image_thresholds[thr_idx],binarise=True,zero_below=zero_below)
				else:
					mask_data = np.multiply(mask_data,threshold_image(contrast_image_list[thr_idx],threshold=image_thresholds[thr_idx],binarise=True,zero_below=zero_below))
	mask_ids = np.unique(mask_data)
	mask_ids.sort()

	#pre-allocate array and then fill with data as we get it
	data_matrix = np.zeros((len(contrast_image_list),np.sum(mask_data.flatten()>0)))*np.nan
	mask_id_start_stop  = np.zeros((np.sum(mask_ids>0),3))*np.nan

	for image_idx, contrast_image in enumerate(contrast_image_list):
		start = 0 #to keep track of indices for start and stop of mask_ids when segmentation provided

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
	return {'data_matrix_full':data_matrix, 'contrast_names': contrast_image_list, 'mask_id_start_stop':mask_id_start_stop.astype(int)}

def _run_lm(d,formula,el,dframe,colname_head,output_vars,res_t,res_p,res_rsquared_adj):
	'''
	For parallelisation of statsmodels call to statsmodels.formula.api.ols .
	Requires memmapped input/output data.
	Places results in global variables, returns nothing
	'''
	import statsmodels.formula.api as smf
	vdata = np.transpose(np.squeeze(d[:,el,:]))
	dframe[dframe.columns[dframe.columns.str.startswith(colname_head)]] = vdata #put the data where the contrast_images were
	lmf = smf.ols(formula=formula,data=dframe).fit()
	for output_var_idx, output_var in enumerate(output_vars):
		res_p[output_var_idx,el] = lmf.pvalues[output_var]
		res_t[output_var_idx,el] = lmf.tvalues[output_var]
	res_rsquared_adj[el] = lmf.rsquared_adj

#INITIAL TRY: NOT TESTED
def _run_mixedlm(d,formula,el,dframe,colname_head,output_vars,res_t,res_p,res_rsquared_adj,**kwargs):
	'''
	For parallelisation of statsmodels call to statsmodels.formula.api.ols .
	Requires memmapped input/output data.
	Places results in global variables, returns nothing
	'''
	import statsmodels.formula.api as smf
	vdata = np.transpose(np.squeeze(d[:,el,:]))
	dframe[dframe.columns[dframe.columns.str.startswith(colname_head)]] = vdata #put the data where the contrast_images were
	lmf = smf.mixedlm(formula=formula,data=dframe,**kwargs).fit()
	for output_var_idx, output_var in enumerate(output_vars):
		res_p[output_var_idx,el] = lmf.pvalues[output_var]
		res_t[output_var_idx,el] = lmf.tvalues[output_var]

def element_lm(data_matrix_full,descriptives,formula,output_vars,contrast_images_colname_head='img_',n_procs=1,tmp_folder=None,**kwargs):
	'''
	Element-wise OLS linear model using statsmodels.formula.api.ols . Will correctly treat
	from 1-3 dimensions if 0th dim is always contrast_images, 1st dim is always elements,
	2nd always subjects. Works equally well with vectors of data, data from image
	volumes, and data from vertices.
	Adding the kwarg groups calls a mixedlm.

	data_matrix_full: np.ndarray
		Full matrix of data from each subject and contrast provided.
			0th dim: contrast images
			1st dim: data elements (voxels, vertices)
			2nd dim: subjects (or timepoints, or both)
	descriptives: csv|pd.DataFrame
		.csv file or pandas dataframe containing:
			1) full path(s) to subject contrast images (1 or more)
			2) additional demographic, group, or control variables for analyses
	formula: str
		Written linear model of the form 'Y ~ X + Z'. Including
		groups='groupingVar' in **kwargs runs a mixedlm rather than ols.
	output_vars: str|list
		Variables that are of interest for output maps (tvalues/pvalues)
		Intercept will automatically be included in the output, do not add it.
	contrast_images_colname_head: str
		Unique header text that is used in the descriptives file to label the
		columns containing the full path to the image files. A simple match will
		be performed to identify all columns that start with this string.
	n_procs: int {1}
		Number of processes to use for linear model parallel processing. Uses
		joblib on local machine
	tmp_folder: str {None}
		Full path to existing folder for dumping the memmap files during
		parallel processing
	kwargs: **kwargs
		Additional arguments for passing to the lm / mixed_lm. Passing
		groups='groupingVar' performs a linear mixed model. See help in
		statsmodels.formula.api. ols|mixedlm

	Returns: dict
		Dictionary of results from lm | mixedlm including {'pvalues','tvalues',
		'rsquared_adj','variable_names'}. rsquared_adj filled with 0s for
		mixedlm, as r2 is undefined.
	'''
	import statsmodels.formula.api as smf
	from sys import stdout as stdout
	import shutil
	import os
	import time
	import tempfile
	from joblib import Parallel, delayed, load, dump

	start_t = time.time()
	if 'groups' in kwargs:
		mixed_model = True
	else:
		mixed_model = False

	#quick attempt to make the input matrix well behaved if we only have 1d or 2d data (1d should work fine, 2d likely not)
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

	res_p = np.zeros((len(output_vars),data_matrix_full.shape[1]))*np.nan #each output var gets its own row
	res_t = np.copy(res_p)*np.nan
	res_rsquared_adj = np.zeros((data_matrix_full.shape[1]))*np.nan #single row of R2, only one per model
	#print(res_p.shape)

	# this is likely quite slow, since we run linear models separatenly for each element :-/
	if n_procs == 1:
		for el_idx in range(data_matrix_full.shape[1]):
			vdata = np.transpose(np.squeeze(data_matrix_full[:,el_idx,:]))
			df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]] = vdata #put the data where the contrast_images were
			if not mixed_model:
				lmf = smf.ols(formula=formula,data=df).fit()
			else:
				lmf = smf.mixedlm(formula=formula,data=df,**kwargs).fit()
			for output_var_idx, output_var in enumerate(output_vars):
				res_p[output_var_idx,el_idx] = lmf.pvalues[output_var]
				res_t[output_var_idx,el_idx] = lmf.tvalues[output_var]
			if not mixed_model:
				res_rsquared_adj[el_idx] = lmf.rsquared_adj
			#progress = (el_idx + 1) / len(data_matrix_full.shape[1])
			#print(" Processed: {0}/{1}".format(el_idx,data_matrix_full.shape[1]),end='\r')
	        #stdout.write(" Processed: {0}/{1} {2}".format(el_idx,data_matrix_full.shape[1],"\r"))
	        #stdout.flush()
	else:
		#memmap the arrays so that we can write to them from all processes
		#as shown here: https://pythonhosted.org/joblib/parallel.html#writing-parallel-computation-results-in-shared-memory
		if tmp_folder is None:
			tmp_folder = tempfile.mkdtemp()
		try:
			res_t_name = os.path.join(tmp_folder,'res_t')
			res_p_name = os.path.join(tmp_folder,'res_p')
			res_rsq_name = os.path.join(tmp_folder,'res_rsq')
			res_t = np.memmap(res_t_name, dtype=res_t.dtype,shape=res_t.shape,mode='w+')
			res_p = np.memmap(res_p_name, dtype=res_p.dtype,shape=res_p.shape,mode='w+')
			res_rsquared_adj = np.memmap(res_rsq_name, dtype=res_rsquared_adj.dtype,shape=res_rsquared_adj.shape,mode='w+')
			dump(data_matrix_full, os.path.join(tmp_folder,'data_matrix_full'))
			data_matrix_full = load(os.path.join(tmp_folder,'data_matrix_full'),mmap_mode='r')
			dump(df,os.path.join(tmp_folder,'descriptives_df'))
			df = load(os.path.join(tmp_folder,'descriptives_df'),mmap_mode='r')
		except:
			print('Could not create memmap files, make sure that {0} exists'.format(tmp_folder))
		print('Memmapping input data and results outputs for parallelisation to {0}'.format(tmp_folder))
		#now parallelise for speeds beyond your wildest imagination, thanks Gael!
		if not mixed_model:
			Parallel(n_jobs=n_procs)(delayed(_run_lm)(data_matrix_full,formula,el_idx,df,contrast_images_colname_head,output_vars,res_t,res_p,res_rsquared_adj)
									for el_idx in range(data_matrix_full.shape[1]))
		else:
			Parallel(n_jobs=n_procs)(delayed(_run_lm)(data_matrix_full,formula,el_idx,df,contrast_images_colname_head,output_vars,res_t,res_p,res_rsquared_adj,**kwargs)
									for el_idx in range(data_matrix_full.shape[1]))

	res = {}
	res['tvalues'] = np.copy(res_t)
	res['pvalues'] = np.copy(res_p)
	res['rsquared_adj'] = np.copy(res_rsquared_adj)
	res['variable_names'] = output_vars

	#cleanup our mess
	if n_procs is not 1:
		try:
			shutil.rmtree(tmp_folder)
		except:
			print("Failed to delete: {0}".format(tmp_folder))
	end_t = time.time()
	print("Performed {0} linear models with data from {1} subjects/timepoints in {2:.2f}s.".format(data_matrix_full.shape[1],data_matrix_full.shape[2],end_t-start_t))
	return res

def plot_stats_single_element(data_group,coordinate,contrasts_plotting_idxs=[0,1],mask_file=None,coordinate_in_mm=False,suppress_plotting=False,alpha=0.2):
	'''

	'''
	#TODO: scale to plot ROI averages using mask_id_start_stop
	data_matrix_full = data_group['data_matrix_full']
	contrast_names = data_group['contrast_names']
	mask_id_start_stop = data_group['mask_id_start_stop']

	if mask_file is not None:
		mask_img = load_volume(mask_file)
		mask_d = mask_img.get_data()
		aff = mask_img.get_affine()
		if coordinate_in_mm:
			coordinate = _mm2vox(aff,coordinate)

		#find where the coordinates are in the flattened volume
		tmp_idx = np.max(mask_d)+1
		if mask_d[coordinate] == 0:
			print('The coordinate that you have chosen is outside of your mask, this only works if you choose a location for which you have data...')
			return None
		mask_d[coordinate] = tmp_idx
		mask_d = mask_d[mask_d>0] #flattened array of masked locations only
		coordinate_1d = np.where(mask_d==tmp_idx)
	else: #no mask file, so we assume that the coordinate is an index to the data_matrix_full (0-based)
		coordinate_1d = coordinate
	if np.ndim(data_matrix_full) == 3: #if we have more than one individual, we only plot the single coordinate's data
		plotting_data = np.squeeze(data_matrix_full[:,coordinate_1d,:])
	else: #we were only passed data from a single individual, so plot all coordinates for the two contrasts
		plotting_data = data_matrix_full
	if not suppress_plotting:
		import matplotlib.pyplot as plt
		plt.plot(plotting_data[contrasts_plotting_idxs[0],:],plotting_data[contrasts_plotting_idxs[1],:],'.',alpha=alpha)
		plt.xlabel(os.path.basename(contrast_names[contrasts_plotting_idxs[0]]))
		plt.ylabel(os.path.basename(contrast_names[contrasts_plotting_idxs[1]]))
		plt.show()
	return plotting_data

def extract_data_group(descriptives,contrast_images_colname_head='img_',mask_file=None,fill_na=0):
	'''
	Extract data from multiple subjects with one or more different contrast
	images. Automatically recognises and pulls multiple contrast images if
	present, with column names starting with contrast_images_colname_head.

	descriptives: csv|pd.DataFrame
		.csv file or pandas dataframe containing:
			1) full path(s) to subject contrast images|files (1 or more)
			2) additional demographic, group, or control variables for analyses
	contrast_images_colname_head: str
		Unique text for name(s) of column(s) where subject contrast images are
		listed. e.g. contrast_image_ for: ['contrast_image_FA',contrast_image_T1']

    Returns: dict
        data_matrix:
        	['data_matrix_full'] - data matrix for this individual
        	['contrast_names'] - filenames
        	['mask_id_start_stop'] - label idx, start, and stop indices in data_matrix_full for non-binary segmentation
        ['data_matrix_full']:
    		0th dim: contrast images
    		1st dim: data elements (voxels)
    		2nd dim: subjects (or timepoints, or both)
	'''
	import time
	start_t = time.time()

	if isinstance(descriptives,basestring): #should be a csv file, load it
		df = pd.read_csv(descriptives,header=0)
	elif isinstance(descriptives, pd.DataFrame): #check if this is a pd.DataFrame
		df = descriptives

	df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
	contrasts_list = df_contrasts_list.values.tolist()

	for contrasts_idx,contrasts in enumerate(contrasts_list): #TODO: this will fail with only a single individual's data
		res = extract_data_multi_image(contrasts,mask_file=mask_file,image_thresholds=None) #no image thresh in group data, all data needs to be the same so use a mask
		data_matrix = res['data_matrix_full']

		#if this is the first time through, we use the shape of the data_matrix to set our output size
		if contrasts_idx == 0:
			data_matrix_full = np.zeros((data_matrix.shape + (np.shape(contrasts_list)[0],)))
			mask_id_start_stop = res['mask_id_start_stop']
		data_matrix_full[:,:,contrasts_idx] = data_matrix
	end_t = time.time()
	print("Data matrix of shape {1} (contrast, element, subject) extracted in {0:.2f} secs".format((end_t-start_t),data_matrix_full.shape))
	if fill_na is not None:
		data_matrix_full[np.isnan(data_matrix_full)] = fill_na
	return {'data_matrix_full':data_matrix_full, 'mask_id_start_stop':mask_id_start_stop,'contrast_names':df.columns[df.columns.str.startswith(contrast_images_colname_head)].tolist()}

def write_element_results(res,descriptives,output_dir,file_name_head,contrast_images_colname_head='img_',mask_file=None,fdr_p='bh',alpha=0.05):
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
			#print(out_fname)

			#write the volume for tvals
			out_data[mask] = res['tvalues'][var_idx]
			out_fname = os.path.join(output_dir,file_name_head + '_' + variable + '_t.nii.gz')
			head['cal_max'] = out_data.max()
			head['cal_min'] = out_data.min()
			img = nb.Nifti1Image(out_data,aff,header=head)
			save_volume(out_fname,img)
			#print(out_fname)

			if fdr_p is not None:
				import statsmodels.stats.multitest as mt
				if fdr_p is 'bh_twostage':
					rejected, cor_p, m0, alpha_stages = mt.fdrcorrection_twostage(res['pvalues'][var_idx],alpha=alpha,method='bh',is_sorted=False)
				elif fdr_p is 'bh':
					rejected, cor_p = mt.fdrcorrection(res['pvalues'][var_idx],alpha=alpha,method='indep',is_sorted=False)
				#write the volume for corrected pvals
				out_data[mask] = cor_p
				out_fname = os.path.join(output_dir,file_name_head + '_' + variable + '_fdr_cor_p.nii.gz')
				head['cal_max'] = out_data.max()
				head['cal_min'] = out_data.min()
				img = nb.Nifti1Image(out_data,aff,header=head)
				save_volume(out_fname,img)

				#write the volume for thresholded t-vals
				temp_t = res['tvalues'][var_idx]
				temp_t[~rejected] = 0 #set to 0 when fail to reject null
				out_data[mask] = temp_t
				out_fname = os.path.join(output_dir,file_name_head + '_' + variable + '_fdr_cor_t.nii.gz')
				head['cal_max'] = out_data.max()
				head['cal_min'] = out_data.min()
				img = nb.Nifti1Image(out_data,aff,header=head)
				save_volume(out_fname,img)

		#write the r2 volume
		out_data[mask] = res['rsquared_adj']
		out_fname = os.path.join(output_dir,file_name_head + '_' + 'model' + '_r2adj.nii.gz')
		head['cal_max'] = out_data.max()
		head['cal_min'] = out_data.min()
		img = nb.Nifti1Image(out_data,aff,header=head)
		save_volume(out_fname,img)
		#print(out_fname)

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
