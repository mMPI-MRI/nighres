from __future__ import print_function
import numpy as np
import pandas as pd
import nibabel as nb
import os

# import os
# import sys
# import cbstools

from ..io import load_volume, save_volume


# from ..utils import _output_dir_4saving
def generate_group_mask(contrast_images, thresholds=None, contrast_images_colname_head='img_',
                        zero_below=True, apply_threshold_to_individual_images = True, verbosity = 0):
    '''
    Generate a binary group mask for the contrast images provided in the dataframe.
    Thresholds are applied to contrast_images in the same order that they exist
    in the dataframe. Produces a binary mask where all conditions are met in all
    input images.
    '''
    if verbosity > 0:
        verbose = True
    else:
        verbose = False

    if isinstance(contrast_images, pd.DataFrame):
        df = contrast_images
    else:
        print('Non-dataframe input is not currently supported')
        return None
    # elif isinstance(contrast_images,list):
    #    df = pd.DataFrame(contrast_images,columns=['img_unknown'])
    #    pass

    # extract the columns with the filenames for each contrast
    df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
    contrasts_list = df_contrasts_list.values.tolist()
    num_cons = df_contrasts_list.shape[1]

    # keep track of the columns that contain the contrast files
    contrast_names = df.columns[df.columns.str.startswith(contrast_images_colname_head)].values
    thresholds_dict = dict(zip(contrast_names, thresholds)) #necessary for correct lookup of thresholds, dict is sorted

    if len(thresholds) != num_cons:
        print(
            'You have not supplied the correct number of threshold values {} for the number of contrasts included in your dataframe {}'.format(
                len(thresholds), num_cons))
        return None

    print('Using contrasts: {} and their matched thresholds: {}'.format(contrast_names,thresholds))
    subject_count = 1
    images_dict = {}

    for contrasts_idx, contrasts in enumerate(contrasts_list):
        if apply_threshold_to_individual_images:
            if contrasts_idx == 0:
                mask_data = threshold_image_list(contrasts, thresholds=thresholds, zero_below=zero_below, verbose=verbose)
            else:
                mask_data = np.multiply(mask_data,
                                        threshold_image_list(contrasts, thresholds=thresholds, zero_below=zero_below,
                                                             verbose=verbose))
        else: #or else we need to create a mean image and then threshold it
            if contrasts_idx == 0: #we need to create the matrix that we will work on by filling it with the first image, for each contrast
                contrast_num = 1
                for contrast in contrasts:
                    if verbosity > 1:
                        print("  Input file: {}".format(contrast))
                    images_dict[contrast_names[contrast_num-1]] = load_volume(contrast).get_data() # -1 to go to 0-indexing
                    contrast_num += 1
            else: #we already loaded one and created the dict, so we just add to them
                contrast_num = 1
                for contrast in contrasts:
                    if verbosity > 1:
                        print("  Input file: {}".format(contrast))
                    images_dict[contrast_names[contrast_num-1]] = np.add(images_dict[contrast_names[contrast_num-1]],
                                                                         load_volume(contrast).get_data())
                    contrast_num += 1
        subject_count += 1
    if not apply_threshold_to_individual_images:
        if verbosity > 0:
            print("Applying threshold to mean image")
        mean_dict = {}
        for key in images_dict: #create the mean images (one for each contrast)
            print("  {} thresholded at {}".format(key,thresholds_dict[key]))
            mean_dict[key] = images_dict[key].astype(np.float) / float(subject_count)

            if zero_below:
                images_dict[key][mean_dict[key]<thresholds_dict[key]] = 0
            else:
                images_dict[key][mean_dict[key]>thresholds_dict[key]] = 0
            images_dict[key] = images_dict[key].astype(bool).astype(int)

    img = load_volume(contrasts[0])
    head = img.get_header()
    aff = img.get_affine()

    #set the ouptut
    if apply_threshold_to_individual_images:
        img = nb.Nifti1Image(mask_data, aff, header=head)
        return img
    else: #make a dictionary for all of the interesting outputs
        for key_idx,key in enumerate(images_dict):
            if key_idx == 0:
                mask_data = images_dict[key]
            else:
                mask_data = np.multiply(mask_data,images_dict[key])
            images_dict[key] = nb.Nifti1Image(images_dict[key],aff,header=head)
            mean_dict[key] = nb.Nifti1Image(mean_dict[key],aff,header=head)
        return {'combined_mask':nb.Nifti1Image(mask_data,aff,header=head),'individual_contrast_masks':images_dict, 'individual_contrast_means':mean_dict}


# TODO: make internal
def threshold_image_list(img_list, thresholds=None, zero_below=True, verbose=True):
    # create a new mask based on the thresholds that were passed
    binarise = True
    if thresholds is None:  # if nothing supplied, use 0
        thresholds = np.zeros(len(img_list))
    if verbose:
        print('Generating combined binary image mask based on supplied threshold(s)')
        print('  Files: \n    {0}'.format(img_list))
        print('  Image threshold values: {0}'.format(thresholds))
    for thr_idx, thresh_val in enumerate(thresholds):
        if thr_idx == 0:
            mask_data = threshold_image(img_list[thr_idx], threshold=thresholds[thr_idx], binarise=True,
                                        zero_below=zero_below)
        else:
            mask_data = np.multiply(mask_data,
                                    threshold_image(img_list[thr_idx], threshold=thresholds[thr_idx], binarise=True,
                                                    zero_below=zero_below))
    return mask_data


def threshold_image(img, threshold=0, binarise=True, zero_below=True):
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

    if isinstance(img, basestring):
        mask_img = load_volume(img)
        data = mask_img.get_data()
    if isinstance(img, np.ndarray):
        data = img
    if zero_below:
        data[data < threshold] = 0
    if not zero_below:
        data[data > threshold] = 0
    if binarise:
        data = data.astype(bool).astype(int)
    return data


def _mm2vox(aff, pts):
    import nibabel as nb
    import numpy as np
    # convert xyz coords from mm to voxel space coords
    return (nb.affines.apply_affine(np.linalg.inv(aff), pts)).astype(int)


def _vox2mm(aff, pts):
    import nibabel as nb
    # convert from voxel coords space back to mm space xyz
    return nb.affines.apply_affine(aff, pts)


def _volume_to_1d(volume_file, mask=None):
    '''
    Converts 3D volume to 1d vector
    '''

    img = load_volume(volume_file)
    data = img.get_data()

    # if there is a mask, convert it to boolean and select only that data
    if mask is not None:
        if isinstance(mask, basestring):
            mask_img = load_volume(mask)
            mask_bool = mask_img.get_data().astype(bool)
            return data[mask_bool]
        elif isinstance(mask, np.ndarray):
            mask_bool = mask.astype(bool)
            return data[mask_bool]
    else:
        return data.flatten()

def element_corrcoef(data_matrix, contrast_idxs=None, contrast1=None, contrast2=None, pvalues=True):
    '''
    Calculate the correlation between two (TODO: or more) contrasts provided in data_matrix format
    :param data_matrix: dict
        dictionary with :
            'data_matrix_full'  - full data matrix of shape contrast by element by subject
            'contrast_names'    - names of image contrasts
            ...                 - others not used
    :param contrast_idxs: list of int
        0-based index of location within the data_matrix['data_matrix_full'] (dimension 0) of the 1st and 2nd contrast.
        If provided, this is used and contrast1/contrast2 are ignored
    :param contrast1: str
        Name of the contrast in the data matrix for the first contrast
    :param contrast2: str
        Name of the contrast in the data matrix for the second contrast
    pvalues: bool
        Output results with pvalues as well as rvalues.
        If True, uses scipy.stats.pearsonr
        If False, numpy.corrcoef is used
    :return:
        res: dict
            Dictionary of results containing 'rvalues' and 'contrast_names' for each element in the
            data_matrix['data_matrix_full'].
            If pvalues=True, also returns 'pvalues' within the dictionary, and field 'calculation' to record what was
            done to obtain the results
    '''
    import time
    start_t = time.time()

    #TODO: make generalizable to all indices - looping!
    data_matrix_full = data_matrix['data_matrix_full']
    contrast_names = data_matrix['contrast_names']

    res = {'contrast_names': contrast_names}

    #base on names if indices not passed
    if contrast_idxs is None:
        contrast_idxs[0] = contrast_names.index(contrast1)
        contrast_idxs[1] = contrast_names.index(contrast2)

    res_r = np.zeros(data_matrix_full.shape[1])*np.nan
    xvar = data_matrix_full[contrast_idxs[0],:,:]
    yvar = data_matrix_full[contrast_idxs[1],:,:]

    if not pvalues:
        for el_idx in range(0,data_matrix_full.shape[1]):
            res_r[el_idx] = np.corrcoef(xvar[el_idx,:],yvar[el_idx,:])[0,1]
        res['rvalues'] = res_r
        res['calculation'] = 'numpy.corrcoef(contrast1,contrast2)'

    else:
        from scipy.stats import pearsonr as pr
        res_p = np.copy(res_r)
        for el_idx in range(0, xvar.shape[0]):
            try:
                res_r[el_idx], res_p[el_idx] = pr(xvar[el_idx,:],yvar[el_idx,:])
            except:
                print(el_idx)
                return res_r, res_p, xvar, yvar
        res['rvalues'] = res_r
        res['pvalues'] = res_p
        res['calculation'] = 'scipy.stats.pearsonr(contrast1,contrast2)'
    end_t = time.time()
    print("Performed {0} correlations with data from {1} subjects/timepoints in {2:.2f}s.".format(
        data_matrix_full.shape[1], data_matrix_full.shape[2], end_t - start_t))
    return res


def basic_element_lm(data_matrix,criterion,predictors_list,descriptives=None,contrast_images_colname_head='img_',add_intercept=True):
    '''

    :param data_matrix:
    :param criterion:
    :param predictors_list:
    :param descriptives:
    :param contrast_images_colname_head:
    :param add_intercept:
    :return:
    '''
    data_matrix_full = data_matrix['data_matrix_full']
    data_vars = data_matrix['contrast_names'] #these are the var names of the contrast images
    res_rsquared_adj = np.zeros((data_matrix_full.shape[1])) * np.nan  # single row of R2, only one per model

    pred_vars_descriptives = []
    pred_vars_data = []
    if add_intercept:
        X = np.zeros(len(predictors_list)+1,data_matrix_full.shape[2])
        X[0,:] = np.ones(data_matrix_full.shape[2]) #constant term first
        X_idx = 1
    else:
        X = np.zeros(len(predictors_list), data_matrix_full.shape[2])
        X_idx = 0

    if descriptives is not None:
        col_names = descriptives.columns.values.tolist()
        for pred in predictors_list:
            if pred in col_names:
                pred_vars_descriptives.append(pred)
                X[X_idx,:] = descriptives[pred].values
                X_idx += 1

    pred_vars_data_idx = [] #this will contain the indices of the data_matrix_full (0th idx) for the var(s) of interest
    for pred in predictors_list:
        if pred in data_vars:
            pred_vars_data.append(pred)
            pred_vars_data_idx.append(data_vars.index(pred))

    # get the criterion variable index if it is in the variables that we have in the data_matrix_full
    # we will have to do the same check below before pulling the data in
    if criterion in data_vars:
        criterion_idx = data_vars.index(criterion)


    for el_idx in range(data_matrix_full.shape[1]):
        if criterion in data_vars:
            Y = np.squeeze(data_matrix_full[criterion_idx,el_idx,:])
        else:
            Y = descriptives[criterion]

        if pred_vars_data_idx: #if we have something here, we need to add its data to the X matrix
            if pred_vars_data_idx:
                X[X_idx,:] = np.squeeze(data_matrix_full[pred_vars_data_idx,el_idx,:]) #pull the data from the big matrix
                X_idx += len(pred_vars_data_idx)
        #now run the linear model in a better way?
        # https://gist.github.com/fabianp/9396204419c7b638d38f

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
        v2[v1 == 0] = np.nan
        v2[v2 == 0] = np.nan
        v1[v1 == 0] = np.nan
        v1[v2 == 0] = np.nan
        v1 = v1[~np.isnan(v1)]
        v2 = v2[~np.isnan(v2)]

    res = {'contrast_image1': contrast_image1, 'contrast_image2': contrast_image2, 'zeros_removed': remove_zero_voxels,
           'corrcoef': np.corrcoef(v1, v2)}
    if return_data_vectors:
        res["input_vectors"] = np.vstack([v1, v2])
    return res


def extract_data_multi_image(contrast_image_list, mask_file=None, image_thresholds=None, zero_below=True,
                             verbose=False):
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
    if isinstance(contrast_image_list, basestring):
        contrast_image_list = [contrast_image_list]

    if mask_file is not None:
        mask_img = load_volume(mask_file)
        mask_data = mask_img.get_data()
    else:
        mask_data = np.ones(load_volume(contrast_image_list[0]).shape)
        if image_thresholds is not None:

            # convert to list if single value passed
            if isinstance(image_thresholds, int) or isinstance(image_thresholds, float):
                image_thresholds = [image_thresholds]
            elif len(image_thresholds) == 1:
                image_thresholds = [image_thresholds]

            # create a new mask based on the thresholds that were passed
            if verbose:
                print('Generating combined binary image mask based on supplied threshold(s)')
                print('  Files: \n    {0}'.format(contrast_image_list))
                print('  Image threshold values: {0}'.format(image_thresholds))
            for thr_idx, thresh_val in enumerate(image_thresholds):
                if thr_idx == 0:
                    mask_data = threshold_image(contrast_image_list[thr_idx], threshold=image_thresholds[thr_idx],
                                                binarise=True, zero_below=zero_below)
                else:
                    mask_data = np.multiply(mask_data, threshold_image(contrast_image_list[thr_idx],
                                                                       threshold=image_thresholds[thr_idx],
                                                                       binarise=True, zero_below=zero_below))
    mask_ids = np.unique(mask_data)
    mask_ids.sort()

    # pre-allocate array and then fill with data as we get it
    data_matrix = np.zeros((len(contrast_image_list), np.sum(mask_data.flatten() > 0))) * np.nan
    mask_id_start_stop = np.zeros((np.sum(mask_ids > 0), 3)) * np.nan

    for image_idx, contrast_image in enumerate(contrast_image_list):
        start = 0  # to keep track of indices for start and stop of mask_ids when segmentation provided

        for mask_id_idx, mask_id in enumerate(mask_ids[mask_ids > 0]):
            # print("mask id: {}".format(mask_id))
            seg_mask = np.zeros_like(mask_data)
            seg_mask[mask_data == mask_id] = 1
            seg_vec = _volume_to_1d(contrast_image, mask=seg_mask)
            # print(seg_vec.shape)
            stop = len(seg_vec) + start
            # print("{} - {}".format(start,stop))
            data_matrix[image_idx, start:stop] = seg_vec

            # construct a lookup for the mask ids and their respective start
            # and stop columns, but only on the first image since they will
            # all be the same
            if image_idx == 0:
                mask_id_start_stop[mask_id_idx] = np.array([mask_id, start, stop])
            start = np.copy(stop)
    return {'data_matrix_full': data_matrix, 'contrast_names': contrast_image_list,
            'mask_id_start_stop': mask_id_start_stop.astype(int)}


def _run_lm(d, formula, el, dframe, colname_head, output_vars, res_t, res_p, res_rsquared_adj, res_f_p):
    '''
    For parallelisation of statsmodels call to statsmodels.formula.api.ols .
    Requires memmapped input/output data.
    Places results in global variables, returns nothing
    '''
    import statsmodels.formula.api as smf
    vdata = np.transpose(np.squeeze(d[:, el, :]))
    dframe[dframe.columns[
        dframe.columns.str.startswith(colname_head)]] = vdata  # put the data where the contrast_images were
    lmf = smf.ols(formula=formula, data=dframe).fit()
    for output_var_idx, output_var in enumerate(output_vars):
        res_p[output_var_idx, el] = lmf.pvalues[output_var]
        res_t[output_var_idx, el] = lmf.tvalues[output_var]
    res_rsquared_adj[el] = lmf.rsquared_adj
    res_f_p[el] = lmf.f_pvalue

# INITIAL TRY: NOT TESTED
# TODO: finish and test, this is more useful than the standard lm since the standard one can be done with many tools...
def _run_mixedlm(d, formula, el, dframe, colname_head, output_vars, res_t, res_p, res_rsquared_adj, res_f_p, **kwargs):
    '''
    For parallelisation of statsmodels call to statsmodels.formula.api.ols .
    Requires memmapped input/output data.
    Places results in global variables, returns nothing
    '''
    import statsmodels.formula.api as smf
    vdata = np.transpose(np.squeeze(d[:, el, :]))
    dframe[dframe.columns[
        dframe.columns.str.startswith(colname_head)]] = vdata  # put the data where the contrast_images were
    lmf = smf.mixedlm(formula=formula, data=dframe, **kwargs).fit()
    for output_var_idx, output_var in enumerate(output_vars):
        res_p[output_var_idx, el] = lmf.pvalues[output_var]
        res_t[output_var_idx, el] = lmf.tvalues[output_var]


def element_lm(data_matrix_full, formula, output_vars, descriptives=None, contrast_images_colname_head='img_',
               demean_data = False, n_procs=1, tmp_folder=None, **kwargs):
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
    demean_data: bool
        0-center the data from each individual and contrast. This does not alter the mean of other data in the dataframes
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
    #from sys import stdout as stdout
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

    # quick attempt to make the input matrix well behaved if we only have 1d or 2d data (1d should work fine, 2d likely not)
    if np.ndim(data_matrix_full) == 1:
        data_matrix_full = data_matrix_full[:, np.newaxis, np.newaxis]
    elif np.ndim(data_matrix_full) == 2:
        data_matrix_full = data_matrix_full[:, :,
                           np.newaxis]  # this may not work for all cases, depending on the assumptions made

    #remove the mean of each subject's data from each contrast
    if demean_data:
        data_matrix_full = data_matrix_full - np.mean(data_matrix_full,axis=1)[:,np.newaxis,:]

    if isinstance(output_vars, basestring):
        output_vars = [output_vars]
    output_vars.append('Intercept')  # add the intercept for output too

    if isinstance(descriptives, basestring):  # should be a csv file, load it
        df = pd.read_csv(descriptives, header=0)
    elif isinstance(descriptives, pd.DataFrame):
        df = descriptives.copy()

    res_p = np.zeros((len(output_vars), data_matrix_full.shape[1])) * np.nan  # each output var gets its own row
    res_t = np.copy(res_p) * np.nan
    res_rsquared_adj = np.zeros((data_matrix_full.shape[1])) * np.nan  # single row of R2, only one per model
    res_f_p = np.zeros_like(res_rsquared_adj) * np.nan

    # this is likely quite slow, since we run linear models separatenly for each element :-/
    if n_procs == 1:
        for el_idx in range(data_matrix_full.shape[1]):
            vdata = np.transpose(np.squeeze(data_matrix_full[:, el_idx, :]))
            df[df.columns[df.columns.str.startswith(
                contrast_images_colname_head)]] = vdata  # put the data where the contrast_images were
            if not mixed_model:
                lmf = smf.ols(formula=formula, data=df).fit()
            else:
                lmf = smf.mixedlm(formula=formula, data=df, **kwargs).fit()
            for output_var_idx, output_var in enumerate(output_vars):
                res_p[output_var_idx, el_idx] = lmf.pvalues[output_var]
                res_t[output_var_idx, el_idx] = lmf.tvalues[output_var]
            if not mixed_model:
                res_rsquared_adj[el_idx] = lmf.rsquared_adj
            # progress = (el_idx + 1) / len(data_matrix_full.shape[1])
            # print(" Processed: {0}/{1}".format(el_idx,data_matrix_full.shape[1]),end='\r')
            # stdout.write(" Processed: {0}/{1} {2}".format(el_idx,data_matrix_full.shape[1],"\r"))
            # stdout.flush()
    else:
        # memmap the arrays so that we can write to them from all processes
        # as shown here: https://pythonhosted.org/joblib/parallel.html#writing-parallel-computation-results-in-shared-memory
        if tmp_folder is None:
            tmp_folder = tempfile.mkdtemp()
        try:
            res_t_name = os.path.join(tmp_folder, 'res_t')
            res_p_name = os.path.join(tmp_folder, 'res_p')
            res_rsq_name = os.path.join(tmp_folder, 'res_rsq')
            res_f_p_name = os.path.join(tmp_folder, 'res_f_p')

            res_t = np.memmap(res_t_name, dtype=res_t.dtype, shape=res_t.shape, mode='w+')
            res_p = np.memmap(res_p_name, dtype=res_p.dtype, shape=res_p.shape, mode='w+')
            res_rsquared_adj = np.memmap(res_rsq_name, dtype=res_rsquared_adj.dtype, shape=res_rsquared_adj.shape,
                                         mode='w+')
            res_f_p = np.memmap(res_f_p_name, dtype=res_rsquared_adj.dtype, shape=res_rsquared_adj.shape,
                                mode='w+')
            dump(data_matrix_full, os.path.join(tmp_folder, 'data_matrix_full'))
            data_matrix_full = load(os.path.join(tmp_folder, 'data_matrix_full'), mmap_mode='r')
            dump(df, os.path.join(tmp_folder, 'descriptives_df'))
            df = load(os.path.join(tmp_folder, 'descriptives_df'), mmap_mode='r')
        except:
            print('Could not create memmap files, make sure that {0} exists'.format(tmp_folder))
        print('Memmapping input data and results outputs for parallelisation to {0}'.format(tmp_folder))
        # now parallelise for speeds beyond your wildest imagination, thanks Gael!
        if not mixed_model:
            Parallel(n_jobs=n_procs)(
                delayed(_run_lm)(data_matrix_full, formula, el_idx, df, contrast_images_colname_head, output_vars,
                                 res_t, res_p, res_rsquared_adj, res_f_p)
                for el_idx in range(data_matrix_full.shape[1]))
        else:
            Parallel(n_jobs=n_procs)(
                delayed(_run_lm)(data_matrix_full, formula, el_idx, df, contrast_images_colname_head, output_vars,
                                 res_t, res_p, res_rsquared_adj, res_f_p, **kwargs)
                for el_idx in range(data_matrix_full.shape[1]))

    res = {}
    res['tvalues'] = np.copy(res_t)
    res['pvalues'] = np.copy(res_p)
    res['rsquared_adj'] = np.copy(res_rsquared_adj)
    res['model_f_pvalues'] = np.copy(res_f_p)
    res['variable_names'] = output_vars
    res['descriptives'] = descriptives

    # cleanup our mess
    if n_procs is not 1:
        try:
            shutil.rmtree(tmp_folder)
        except:
            print("Failed to delete: {0}".format(tmp_folder))
    end_t = time.time()
    print("Performed {0} linear models with data from {1} subjects/timepoints in {2:.2f}s.".format(
        data_matrix_full.shape[1], data_matrix_full.shape[2], end_t - start_t))

    return res



def plot_stats_single_element(data_group, coordinate, contrasts_plotting_idxs=[0, 1], descriptives_col=None,
                              mask_file=None, coordinate_in_mm=False, suppress_plotting=False, alpha=0.2):
    '''

    #final var in descriptives col is used as Y variable, others as covariates
    '''
    # TODO: scale to plot ROI averages using mask_id_start_stop
    data_matrix_full = data_group['data_matrix_full']
    contrast_names = data_group['contrast_names']
    mask_id_start_stop = data_group['mask_id_start_stop']
    if descriptives_col: #colname(s) selected
        if 'descriptives' in data_group:
            descriptives = data_group['descriptives']
        else:
            print('The data dictionary that you passed does not include descriptives (i.e., a DataFrame')
        Yvar = descriptives[descriptives_col.pop()]
        plots_count = len(contrasts_plotting_idxs)
        if descriptives_col: #if we still have descriptives, we use them as covariates
            Covars = descriptives[descriptives_col]
        else:
            Covars = None
    else:
        plots_count = len(contrasts_plotting_idxs)
        Yvar = None
    #TODO: remove the covariates and plot the residuals

    if mask_file is not None:
        mask_img = load_volume(mask_file)
        mask_d = mask_img.get_data()
        aff = mask_img.get_affine()
        if coordinate_in_mm:
            coordinate = _mm2vox(aff, coordinate)

        # find where the coordinates are in the flattened volume
        tmp_idx = np.max(mask_d) + 1
        if mask_d[coordinate] == 0:
            print(
                'The coordinate that you have chosen is outside of your mask, this only works if you choose a location for which you have data...')
            return None
        mask_d[coordinate] = tmp_idx
        mask_d = mask_d[mask_d > 0]  # flattened array of masked locations only
        coordinate_1d = np.where(mask_d == tmp_idx)
    else:  # no mask file, so we assume that the coordinate is an index to the data_matrix_full (0-based)
        coordinate_1d = coordinate
    if np.ndim(data_matrix_full) == 3:  # if we have more than one individual, we only plot the single coordinate's data
        plotting_data = np.squeeze(data_matrix_full[:, coordinate_1d, :])
    else:  # we were only passed data from a single individual, so plot all coordinates for the two contrasts
        plotting_data = data_matrix_full
    if not suppress_plotting:
        import matplotlib.pyplot as plt
        if Yvar is not None:  # if we already defined the Yvariable with the descriptives, then we plot it versus all of the contrast types
            ydata = Yvar
            #TODO: add for loop here using itertools.permutation (and removing duplicates)
            ylabel = Yvar.name
            xdata = plotting_data[contrasts_plotting_idxs[0]]
            xlabel = os.path.basename(contrast_names[contrasts_plotting_idxs[0]])
        else: # or we set the y to be the next plotting index (plotting all against 1st)
            xdata = plotting_data[contrasts_plotting_idxs[0],:]
            ydata = plotting_data[contrasts_plotting_idxs[1],:]
            xlabel = os.path.basename(contrast_names[contrasts_plotting_idxs[0]])
            ylabel = os.path.basename(contrast_names[contrasts_plotting_idxs[1]])
        f,ax = plt.subplots(figsize=(10,10))
        ax.plot(xdata, ydata, '.', alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.show()
    return f,ax


def extract_data_group(descriptives, contrast_images_colname_head='img_', mask_file=None, fill_na=0):
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
            ['descriptives'] - included dataframe
        ['data_matrix_full']:
            0th dim: contrast images
            1st dim: data elements (voxels)
            2nd dim: subjects (or timepoints, or both)
    '''
    import time
    start_t = time.time()

    if isinstance(descriptives, basestring):  # should be a csv file, load it
        df = pd.read_csv(descriptives, header=0)
    elif isinstance(descriptives, pd.DataFrame):  # check if this is a pd.DataFrame
        df = descriptives

    df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
    contrasts_list = df_contrasts_list.values.tolist()

    for contrasts_idx, contrasts in enumerate(
            contrasts_list):  # TODO: this will fail with only a single individual's data
        res = extract_data_multi_image(contrasts, mask_file=mask_file,
                                       image_thresholds=None)  # no image thresh in group data, all data needs to be the same so use a mask
        data_matrix = res['data_matrix_full']

        # if this is the first time through, we use the shape of the data_matrix to set our output size
        if contrasts_idx == 0:
            data_matrix_full = np.zeros((data_matrix.shape + (np.shape(contrasts_list)[0],)))
            mask_id_start_stop = res['mask_id_start_stop']
        data_matrix_full[:, :, contrasts_idx] = data_matrix
    end_t = time.time()
    print("Data matrix of shape {1} (contrast, element, subject) extracted in {0:.2f} secs".format((end_t - start_t),
                                                                                                   data_matrix_full.shape))
    if fill_na is not None:
        data_matrix_full[np.isnan(data_matrix_full)] = fill_na
    return {'data_matrix_full': data_matrix_full, 'mask_id_start_stop': mask_id_start_stop,
            'contrast_names': df.columns[df.columns.str.startswith(contrast_images_colname_head)].tolist(),
            'descriptives': descriptives}


def write_element_results(res, descriptives, output_dir, file_name_head, contrast_images_colname_head='img_',
                          mask_file=None, fdr_p='bh', alpha=0.05):
    '''
    Write statistical results (pvals,tvals,rsquared_adj) back to the type of file
    from which they were generated.
    '''
    import os
    if fdr_p is not None:
        import statsmodels.stats.multitest as mt

    if isinstance(descriptives, basestring):  # should be a csv file, load it
        df = pd.read_csv(descriptives, header=0)
    elif isinstance(descriptives, pd.DataFrame):
        df = descriptives

    # determine type of file
    df_contrasts_list = df[df.columns[df.columns.str.startswith(contrast_images_colname_head)]]
    contrasts_list = df_contrasts_list.values.tolist()
    # return df_contrasts_list
    if isinstance(contrasts_list, basestring):
        fname = contrasts_list[0]
    else:
        fname = contrasts_list[0][0]
    ext = os.path.basename(fname).split('.', 1)

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
            # write the volume for pvals
            out_data[mask] = res['pvalues'][var_idx]
            out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_p.nii.gz')
            head['cal_max'] = out_data.max()
            head['cal_min'] = out_data.min()
            img = nb.Nifti1Image(out_data, aff, header=head)
            save_volume(out_fname, img)
            # print(out_fname)

            # write the volume for tvals
            out_data[mask] = res['tvalues'][var_idx]
            out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_t.nii.gz')
            head['cal_max'] = out_data.max()
            head['cal_min'] = out_data.min()
            img = nb.Nifti1Image(out_data, aff, header=head)
            save_volume(out_fname, img)
            # print(out_fname)

            if fdr_p is not None:
                if fdr_p is 'bh_twostage':
                    rejected, cor_p, m0, alpha_stages = mt.fdrcorrection_twostage(res['pvalues'][var_idx], alpha=alpha,
                                                                                  method='bh', is_sorted=False)
                elif fdr_p is 'bh':
                    rejected, cor_p = mt.fdrcorrection(res['pvalues'][var_idx], alpha=alpha, method='indep',
                                                       is_sorted=False)
                # write the volume for corrected pvals
                out_data[mask] = cor_p
                out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_p_fdr_cor.nii.gz')
                head['cal_max'] = out_data.max()
                head['cal_min'] = out_data.min()
                img = nb.Nifti1Image(out_data, aff, header=head)
                save_volume(out_fname, img)

                # write the volume for thresholded t-vals
                temp_t = res['tvalues'][var_idx]
                temp_t[~rejected] = 0  # set to 0 when fail to reject null
                out_data[mask] = temp_t
                out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_t_fdr_cor.nii.gz')
                head['cal_max'] = out_data.max()
                head['cal_min'] = out_data.min()
                img = nb.Nifti1Image(out_data, aff, header=head)
                save_volume(out_fname, img)

        # write the r2 volume
        out_data[mask] = res['rsquared_adj']
        out_fname = os.path.join(output_dir, file_name_head + '_' + 'model' + '_r2adj.nii.gz')
        head['cal_max'] = out_data.max()
        head['cal_min'] = out_data.min()
        img = nb.Nifti1Image(out_data, aff, header=head)
        save_volume(out_fname, img)

        # write the model_f_pvalues volume
        out_data[mask] = res['model_f_pvalues']
        out_fname = os.path.join(output_dir, file_name_head + '_' + 'model' + '_f_p.nii.gz')
        head['cal_max'] = out_data.max()
        head['cal_min'] = out_data.min()
        img = nb.Nifti1Image(out_data, aff, header=head)
        save_volume(out_fname, img)

        if fdr_p is not None:
            if fdr_p is 'bh_twostage':
                f_rejected, f_cor_p, f_m0, f_alpha_stages = mt.fdrcorrection_twostage(res['model_f_pvalues'],
                                                                                      alpha=alpha,method='bh',
                                                                                      is_sorted=False)
            elif fdr_p is 'bh':
                f_rejected, f_cor_p = mt.fdrcorrection(res['model_f_pvalues'], alpha=alpha, method='indep',
                                                       is_sorted=False)
#            print(sum(~f_rejected))
#            print(sum(f_cor_p<0.05))
            # write the volume for corrected model f pvals
            out_data[mask] = f_cor_p
            out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_model_f_p_fdr_cor.nii.gz')
            head['cal_max'] = out_data.max()
            head['cal_min'] = out_data.min()
            img = nb.Nifti1Image(out_data, aff, header=head)
            save_volume(out_fname, img)

            #output the 1-p value for easy display (corrected)
            out_data[mask] = 1 - out_data[mask]
            out_fname = os.path.join(output_dir, file_name_head + '_' + 'model' + '_f_1mp_fdr_cor.nii.gz')
            head['cal_max'] = out_data.max()
            head['cal_min'] = out_data.min()
            img = nb.Nifti1Image(out_data, aff, header=head)
            save_volume(out_fname, img)

            # write the volume for thresholded rsqr_adj values
            temp_t = res['rsquared_adj']
            temp_t[~f_rejected] = 0  # set to 0 when fail to reject null
            out_data[mask] = temp_t
            out_fname = os.path.join(output_dir, file_name_head + '_' + variable + '_r2adj_fdr_cor.nii.gz')
            head['cal_max'] = out_data.max()
            head['cal_min'] = out_data.min()
            img = nb.Nifti1Image(out_data, aff, header=head)
            save_volume(out_fname, img)

    elif ext is 'txt':  # working with vertex files
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
        # do stuff here, only because you can output summary stats, otherwise why are you breaking this up into different segments... since it is all voxel-wise anyways
        # TODO: reconsider
        pass
    return None
