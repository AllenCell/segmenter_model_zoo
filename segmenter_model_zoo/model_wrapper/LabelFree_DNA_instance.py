import pandas as pd
import numpy as np
import os
from os import listdir
from aicsimageio import AICSImage, omeTifWriter
import math
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from skimage.morphology import ball, dilation, erosion, disk, binary_closing, skeletonize, skeletonize_3d, watershed, remove_small_objects
from skimage.measure import regionprops, label
from os.path import isfile, join, exists, basename
from scipy import stats
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.segmentation import relabel_sequential, find_boundaries
from aicssegmentation.core.utils import hole_filling
import random 
import pandas as pd 
from tifffile import imsave
from shutil import copyfile 
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
import glob 
from aicsmlsegment.utils import background_sub, simple_norm
from aicsimageio import AICSImage
import itk

mem_pre_cut_th = 0.15
dna_mask_bw_th = 0.5 #1.45
min_dna = 4000 

selem_top = np.zeros((5,5,5),dtype=np.uint8)
selem_top[:3,2,2]=1

def getLargestCC(labels):
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def SegModule(img=None, model_list=None, prune_border=False, filename=None, index=None, return_prediction=False):

    # model order: dna_mask, cellmask
    # img: Z x Y x X

    if img is None:
        # load the image 
        reader = AICSImage(filename)
        img = reader.data[0,index,:,:,:]

    img = img[0,:,:,:]
    # make sure the image has 3 dimensions
    assert len(img.shape)==3

    # model 1: dna_mask
    dna_mask_pred = model_list[0].apply_on_single_zstack(input_img=img)
    dna_mask_bw = dna_mask_pred > dna_mask_bw_th

    #imsave('dna_pred.tiff',dna_mask_pred)

    # model 2: cell edge
    mem_pred = model_list[1].apply_on_single_zstack(input_img=img)

    #imsave('mem_pred.tiff',mem_pred)

    # prepare separation boundary
    tmp_mem = mem_pred>mem_pre_cut_th
    for zz in range(tmp_mem.shape[0]):
        if np.any(tmp_mem[zz,:,:]):
            tmp_mem[zz,:,:] = dilation(tmp_mem[zz,:,:], selem=disk(1))

    # cut DNA
    dna_mask_bw[tmp_mem>0]=0

    # clean up DNA 
    # HACK: Assume mitotic is not important
    dna_mask_bw = remove_small_objects(dna_mask_bw, min_size=min_dna)

    dna_mask_bw = hole_filling(dna_mask_bw, 1, 4000, fill_2d=True)

    dna_label = label(dna_mask_bw, connectivity=1)

    return dna_label
