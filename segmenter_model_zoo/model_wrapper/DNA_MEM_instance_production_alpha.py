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

mem_pre_cut_th = 0.2  # 0.02
seed_bw_th = 0.90
dna_mask_bw_th = 0.5  #0.7 
min_seed_size = 3800 #9000 # 3800

selem_top = np.zeros((5,5,5),dtype=np.uint8)
selem_top[:3,2,2]=1

def getLargestCC(labels):
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def SegModule(img=None, model_list=None, prune_border=False, filename=None, index=None, return_prediction=False):

    # model order: dna_mask, cellmask, dna_seed

    if img is None:
        # load the image 
        reader = AICSImage(filename)
        img = reader.data[0,index,:,:,:]
    
    # make sure the image has 4 dimensions
    assert len(img.shape)==4 and img.shape[0]==2 

    # input channel order:
    # first = dna; second = cell mask

    # extract dna channel 
    dna_img = img[0,:,:,:]
    dna_img[dna_img>60000] = dna_img.min()
    dna_img = background_sub(dna_img,50)
    dna_img = simple_norm(dna_img, 2.5, 10)

    # extra cellmask channel
    mem_img = img[1,:,:,:]
    mem_img[mem_img>60000] = mem_img.min()
    mem_img = background_sub(mem_img,50)
    mem_img = simple_norm(mem_img, 2, 11)

    # model 1: dna_mask
    dna_mask_pred = model_list[0].apply_on_single_zstack(dna_img, already_normalized=True, cutoff=-1)
    dna_mask_bw = dna_mask_pred > dna_mask_bw_th

    #import random
    #rr = random.randint(100,199)
    #imsave('test_dna_mask_'+str(rr)+'.tiff', dna_mask_pred)

    # model 2: cell edge
    mem_pred = model_list[1].apply_on_single_zstack(mem_img, already_normalized=True, cutoff=-1)

    #imsave('test_mem_'+str(rr)+'.tiff', mem_pred)

    # model 3: dna_seed
    seed_pred = model_list[2].apply_on_single_zstack(dna_img, already_normalized=True, cutoff=-1)
    seed_bw = seed_pred>seed_bw_th

    #imsave('test_seed_'+str(rr)+'.tiff', seed_pred)

    # prepare separation boundary
    tmp_mem = mem_pred>mem_pre_cut_th
    for zz in range(tmp_mem.shape[0]):
        if np.any(tmp_mem[zz,:,:]):
            tmp_mem[zz,:,:] = dilation(tmp_mem[zz,:,:], selem=disk(1))

    # cut seed 
    seed_bw[tmp_mem>0]=0

    # find the stack bottom
    stack_bottom = 0
    for zz in np.arange(3,tmp_mem.shape[0]//2):
        if np.count_nonzero(tmp_mem[zz,:,:]>0)>0.5*tmp_mem.shape[1]*tmp_mem.shape[2]:
            stack_bottom = zz
            break

    # find the stack top
    stack_top = mem_pred.shape[0]-1
    for zz in np.arange(mem_pred.shape[0]-1, mem_pred.shape[0]//2+1, -1):
        if np.count_nonzero(tmp_mem[zz,:,:]>0)>64:
            stack_top = zz 
            break

    #print(stack_bottom)
    #print(stack_top)

    # prune mem_pred
    if stack_bottom==0:
        mem_pred[0,:,:]=0.0000001
    else:
        mem_pred[:stack_bottom,:,:]=0.0000001
    mem_pred[stack_top:,:,:]=0.0000001

    #############################################################
    # properly build the seeds
    #############################################################
    # pre-prune the seed
    seed_bw = remove_small_objects(seed_bw, min_size=20)
    seed_label = label(seed_bw, connectivity=1)

    # save the seeds that are touching boundary
    boundary_mask = np.zeros_like(seed_bw)
    boundary_mask[:, :4, :] = 1
    boundary_mask[:,-4:, :] = 1
    boundary_mask[:, :, :4] = 1
    boundary_mask[:, :,-4:] = 1
    
    bd_seed_on_hold = np.zeros_like(seed_bw)
    bd_idx = list(np.unique(seed_label[boundary_mask>0]))
    for index, cid in enumerate(bd_idx):
        if cid>0:
            bd_seed_on_hold[seed_label==cid]=1

    seed_bw = remove_small_objects(seed_bw, min_size=min_seed_size, connectivity=1)

    # finalize seed (add back the seeds on hold)
    seed_bw[bd_seed_on_hold>0]=1
    seed_label, seed_num = label(seed_bw, return_num=True, connectivity=1)
        
    if stack_bottom==0:
        seed_label[0,:,:] = seed_num+1
    else:
        seed_label[:stack_bottom,:,:] = seed_num+1
    seed_label[stack_top:,:,:] = seed_num+2

    #imsave('test_final_seed_'+str(rr)+'.tiff', seed_label)

    ################################################################
    # get cell instance segmentation
    ################################################################
    #cell_seg = watershed(mem_pred, seed_label, watershed_line=True)
    raw0 = mem_pred.astype(np.float32)
    raw_itk = itk.GetImageFromArray(raw0)
    seed_itk= itk.GetImageFromArray(seed_label.astype(np.int16))
    seg_itk = itk.morphological_watershed_from_markers_image_filter(raw_itk, marker_image=seed_itk, fully_connected=True, mark_watershed_line=False)
    cell_seg = itk.GetArrayFromImage(seg_itk)

    #imsave('test_cell_seg_'+str(rr)+'.tiff', cell_seg)

    cell_seg[cell_seg==seed_num+1]=0
    cell_seg[cell_seg==seed_num+2]=0

    ################################################################
    # refine cell segmentation near bottom
    ################################################################
    # estimate colony coverage size
    colony_coverage = np.amax(tmp_mem.astype(np.uint8), axis=0)
    colony_coverage_size = np.count_nonzero(colony_coverage.flat>0)
    #print([colony_coverage_size, colony_coverage.shape[0]*colony_coverage.shape[1]])
    
    if stack_bottom ==0:
        step_down_search = 0
        step_down_z = 0
    else:
        step_down_search = stack_bottom-1
        step_down_z = stack_bottom-1

    for zz in np.arange(step_down_search, cell_seg.shape[0]//2):
        if np.count_nonzero(cell_seg[zz,:,:]>0) > 0.8*colony_coverage_size:
            step_down_z = zz
            break 

    if stack_bottom ==0:
        step_down_start = 0
    else:
        step_down_start = stack_bottom - 1
    for zz in np.arange(step_down_start, step_down_z, 1):
        cell_seg[zz,:,:] = cell_seg[step_down_z,:,:]

    # relabel the index in case altered when dumping the bottom
    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))


    # fix top by 1 more up
    # this is wrong, may cause drift (leading to false dna cut) in the middle part of the cell
    #cell_seg = dilation(cell_seg, selem=selem_top)
    #imsave('test_cell_seg_after_fix_top_'+str(rr)+'.tiff', cell_seg)

    ################################################################
    # get dna instance segmentation
    ################################################################
    # make sure dna is not out of membrane
    dna_mask_bw[cell_seg==0]=0

    '''
    ###
    # cannot prune after using mitotic improved segmentation
    ###
    # prune dna mask
    for zz in range(dna_mask_bw.shape[0]):
        if np.any(dna_mask_bw[zz,:,:]):
            dna_holes = dna_mask_bw[zz,:,:]==0
            dna_holes = remove_small_objects(dna_holes, min_size=400)
            dna_mask_bw[zz,:,:] = remove_small_objects(dna_holes==0, min_size=400)
    dna_mask_bw = remove_small_objects(dna_mask_bw, min_size=2500)
    dna_mask_bw = binary_fill_holes(dna_mask_bw>0)
    '''

    # propagate the cell index to dna
    dna_mask_label = np.zeros_like(cell_seg)
    dna_mask_label[dna_mask_bw>0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    # refine dna in each cell by removing small parts touching seperatinon bounary
    num_cell = cell_seg.max()
    for cell_idx in range(num_cell):
        single_dna = dna_mask_label==(cell_idx+1)
        single_dna_label, num_obj = label(single_dna, return_num=True, connectivity=3)
        if num_obj==1:
            continue

        single_mem_bd = find_boundaries(cell_seg==(cell_idx+1), mode='inner')
        bd_idx = list(np.unique(single_dna_label[single_mem_bd>0]))
        if len(bd_idx)>0:
            for list_idx, dna_bd_idx in enumerate(bd_idx):
                if np.count_nonzero(single_dna_label==dna_bd_idx)<500:
                    dna_mask_label[single_dna_label==dna_bd_idx]=0

    '''
    if prune_border:
        ################################################################
        # remove all border-touching cells
        ################################################################
        bd_idx = list(np.unique(cell_seg[boundary_mask>0]))
        for index, cid in enumerate(bd_idx):
            if cid>0:
                cell_seg[cell_seg==cid]=0
                dna_mask_label[dna_mask_label==cid]=0
    '''

    ### HACK:
    seed_label[seed_label==seed_num+1]=0
    seed_label[seed_label==seed_num+2]=0

    if return_prediction:
        return cell_seg, dna_mask_label, [dna_mask_pred, mem_pred, seed_pred]
    else:
        return cell_seg, dna_mask_label, seed_label
