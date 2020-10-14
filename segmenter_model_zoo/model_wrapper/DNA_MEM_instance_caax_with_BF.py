import pandas as pd
import numpy as np
import sys
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
#from collections import Counter
#import pdb

# from cell_detector import detect

mem_pre_cut_th = 0.2 # 0.2  # + 0.25
min_seed_size = 6000 #9000 # 3800

dna_bf_cutoff = 1.05

flat_se = np.zeros((5,5,5),dtype=np.uint8)
flat_se[2,:,:]=1


def getLargestCC(labels, is_label=True):

    if is_label:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        sub_labels = label(labels>0, connectivity=3, return_num=False)
        largestCC = sub_labels == np.argmax(np.bincount(sub_labels.flat)[1:])+1

    return largestCC


def SegModule(img=None, model_list=None, prune_border=False, filename=None, index=None, return_prediction=False, two_camera=False):

    #if two_camera:
    #    mem_bf_cut = 1.9 ### only use lf pred for membrane top and bottom (the seperation in two camera could be wrong)
    #    dna_bf_cutoff = 1.45 ### decrease cutoff after turn tta off
    #else:
    #    mem_bf_cut = 0.25
    #    dna_bf_cutoff = 1.5
    
    
    # model order: dna_mask, cellmask, dna_seed

    if img is None:
        # load the image 
        reader = AICSImage(filename)
        img = reader.data[0,index,:,:,:]
    
    # make sure the image has 4 dimensions
    #assert len(img.shape)==4 and img.shape[0]==3
    if not (len(img.shape)==4 and img.shape[0]==2 and img.shape[1]>=32):
        print('bad data, dimension crashed')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    ###########################################################
    # part 1: prepare data
    ###########################################################
    print(img.shape)
    # input channel order:
    # first = caax; second = bf

    # extra cellmask channel
    mem_img = img[0,:,:,:].copy()
    mem_img[mem_img>60000] = mem_img.min()
    mem_img = background_sub(mem_img,50)
    mem_img = simple_norm(mem_img, 2, 11)

    print('image normalization is done')
    print('applying all DL models ... ...')

    ###########################################################
    # part 2: run predictions
    ###########################################################

    # model 1: cell edge
    mem_pred = model_list[0].apply_on_single_zstack(mem_img, already_normalized=True, cutoff=-1)

    # model 2: dna from bf
    if two_camera:
        dna_bf_pred = model_list[1].apply_on_single_zstack(input_img = img[1,:,:,:], use_tta=False)
    else:
        dna_bf_pred = model_list[1].apply_on_single_zstack(input_img = img[1,:,:,:], use_tta=True)
    
    dna_bf_bw = dna_bf_pred > dna_bf_cutoff

    print('predictions are done.')

    #rr= random.randint(1000,9000)

    ###########################################################
    # part 3: merge bf based prediction into dye based prediction
    ###########################################################
    '''
    # adjust mem_pred by bf
    mem_bf_trust = np.zeros_like(mem_pred)
    mem_bf_trust[mem_bf_pred>mem_bf_cut]=1
    mem_pred = mem_pred + mem_bf_trust*0.1

    mem_bf_trust_1 = dilation(mem_bf_trust>0, selem=flat_se)
    mem_bf_trust_1 = mem_bf_trust_1.astype(np.uint8)
    mem_bf_trust_1[mem_bf_trust_1>0]=1
    mem_pred = mem_pred + mem_bf_trust_1*0.1

    mem_bf_trust_2 = dilation(mem_bf_trust_1>0, selem=flat_se)
    mem_bf_trust_2 = mem_bf_trust_2.astype(np.uint8)
    mem_bf_trust_2[mem_bf_trust_2>0]=1
    mem_pred = mem_pred + mem_bf_trust_2*0.1

    mem_bf_trust_3 = dilation(mem_bf_trust_2>0, selem=flat_se)
    mem_bf_trust_3 = mem_bf_trust_3.astype(np.uint8)
    mem_bf_trust_3[mem_bf_trust_3>0]=1
    mem_pred = mem_pred + mem_bf_trust_3*0.1
    '''

    # prepare separation boundary
    tmp_mem = mem_pred>mem_pre_cut_th
    for zz in range(tmp_mem.shape[0]):
        if np.any(tmp_mem[zz,:,:]):
            tmp_mem[zz,:,:] = dilation(tmp_mem[zz,:,:], selem=disk(1))

    # cut seed 
    seed_bw = dna_bf_bw.copy()
    seed_bw[tmp_mem>0]=0
    #seed_bw[mem_bf_pred>mem_bf_cut_extra_for_seed]=0 
    # # sometimes the boundary signal is week, we need extra strong cut
    # # but, this may falsely cut a lot more. So, the correct way is not 
    # # to apply extra cut. Instead, we should improvement the mem lf model

    #############################################################
    # prune the seeds first
    #############################################################
    # pre-prune the seed
    seed_bw = remove_small_objects(seed_bw, min_size=20)
    seed_label = label(seed_bw, connectivity=1)

    # save a copy to combine with labelfree prediction for rescuing missed seeds
    seed_bw_before_pruning = seed_bw.copy()

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

    ###########################################################
    # part 4: prepare for watershed image
    ###########################################################

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

    # prune mem_pred
    if stack_bottom==0:
        mem_pred[0,:,:]=0.0000001
    else:
        mem_pred[:stack_bottom,:,:]=0.0000001
    mem_pred[stack_top:,:,:]=0.0000001

    #############################################################
    # part 5: prepare for watershed seed
    #############################################################
    seed_label, seed_num = label(seed_bw, return_num=True, connectivity=1)
        
    if stack_bottom==0:
        seed_label[0,:,:] = seed_num+1
    else:
        seed_label[:stack_bottom,:,:] = seed_num+1
    seed_label[stack_top:,:,:] = seed_num+2

    ################################################################
    # part 6: get cell instance segmentation
    ################################################################
    #cell_seg = watershed(mem_pred, seed_label, watershed_line=True)
    raw0 = mem_pred.astype(np.float32)
    raw_itk = itk.GetImageFromArray(raw0)
    seed_itk= itk.GetImageFromArray(seed_label.astype(np.int16))
    seg_itk = itk.morphological_watershed_from_markers_image_filter(raw_itk, marker_image=seed_itk, fully_connected=True, mark_watershed_line=False)
    cell_seg = itk.GetArrayFromImage(seg_itk)

    cell_seg[cell_seg==seed_num+1]=0
    cell_seg[cell_seg==seed_num+2]=0

    print('watershed based cell segmentation is done.')

    ################################################################
    # part 7: refine cell segmentation near bottom
    ################################################################
    # estimate colony coverage size
    colony_coverage = np.amax(tmp_mem.astype(np.uint8), axis=0)
    colony_coverage_size = np.count_nonzero(colony_coverage.flat>0)
    #print([colony_coverage_size, colony_coverage.shape[0]*colony_coverage.shape[1]])
    
    step_down_z = stack_bottom-1
    for zz in np.arange(stack_bottom-1, cell_seg.shape[0]//2):
        if np.count_nonzero(cell_seg[zz,:,:]>0) > 0.8*colony_coverage_size:
            step_down_z = zz
            break 

    for zz in np.arange(stack_bottom, step_down_z, 1):
        cell_seg[zz,:,:] = cell_seg[step_down_z,:,:]

    print('stack bottom has been properly updated.')

    #### remove small cells due to failure / noise
    for ii in np.unique(cell_seg[cell_seg>0]):
        this_one_cell = cell_seg==ii
        this_dna = dna_bf_bw.copy()
        this_dna[this_one_cell==0]=0
        if np.count_nonzero(this_one_cell>0)< 70000 or np.count_nonzero(this_dna>0)< 1000 or np.count_nonzero(getLargestCC(this_dna, is_label=False))< 10000: # small "cell" or "dna"
            cell_seg[this_one_cell>0]=0

    '''
    # false clip check (mem channel)
    z_range_mem = np.where(np.any(cell_seg, axis=(1,2)))
    z_range_mem = z_range_mem[0]
    if len(z_range_mem)==0 or z_range_mem[0] == 0 or z_range_mem[-1] == cell_seg.shape[0] - 1:
        print('exit because false clip or bad floaty is detected in mem channel')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None
    '''


    # relabel the index in case altered when dumping the bottom
    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))

    print('size based QC is done')


    # fix top by 1 more up
    # this is wrong, may cause drift (leading to false dna cut) in the middle part of the cell
    #cell_seg = dilation(cell_seg, selem=selem_top)
    #imsave('test_cell_seg_after_fix_top_'+str(rr)+'.tiff', cell_seg)

    ################################################################
    # get dna instance segmentation
    ################################################################
    # make sure dna is not out of membrane
    dna_bf_bw[cell_seg==0]=0

    # propagate the cell index to dna
    dna_mask_label = np.zeros_like(cell_seg)
    dna_mask_label[dna_bf_bw>0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    # false clip check (dna channel)
    z_range_dna = np.where(np.any(dna_mask_label, axis=(1,2)))
    z_range_dna = z_range_dna[0]
    if len(z_range_dna)==0 or z_range_dna[0] == 0 or z_range_dna[-1] == dna_mask_label.shape[0] - 1:
        print('exit because false clip or bad floaty is detected in dna channel')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None
        #sys.exit(0)

    if dna_mask_label.max()<3:  # if only a few cells left, just throw it away
        print('exit because only very few cells are segmented, maybe a failed image, please check')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None
        #sys.exit(0)

    '''
    print('refining dna masks ... ...')

    # get the index touching border
    bd_idx = list(np.unique(cell_seg[boundary_mask>0]))

    
    # refine dna 
    num_cell = cell_seg.max()
    for cell_idx in range(num_cell):
        if (cell_idx + 1) in bd_idx:
            # no need to refine, because this will be ignored in the real analysis
            continue

        if not np.any(dna_mask_label==(cell_idx+1)):  # empty dna should have been removed
            print('bug, empty dna is found, but should not')
            if return_prediction:
                return None, [dna_mask_pred, mem_pred, seed_pred]
            else:
                return None

        single_dna = dna_mask_label==(cell_idx+1)
        single_dna_label, num_obj = label(single_dna, return_num=True, connectivity=3)
        if num_obj ==1:
            largest_label = single_dna_label>0
        elif num_obj>1:
            largest_label = getLargestCC(single_dna_label)
        else:
            print('bug occurs in processing the pair ... ')
            if return_prediction:
                return None, [dna_mask_pred, mem_pred, seed_pred]
            else:
                return None
        ratio_check = np.count_nonzero( seed_label==(cell_idx+1) ) / np.count_nonzero( largest_label )
        if  ratio_check < 0.75: #interphase
            # prune dna mask
            for zz in range(single_dna.shape[0]):
                if np.any(single_dna[zz,:,:]):
                    single_dna[zz,:,:] = binary_fill_holes(single_dna[zz,:,:])
                    #dna_holes = single_dna[zz,:,:]==0
                    #dna_holes = remove_small_objects(dna_holes, min_size=400)
                    single_dna[zz,:,:] = remove_small_objects(single_dna[zz,:,:], min_size=100)
            single_dna = remove_small_objects(single_dna, min_size=2500)
            dna_mask_label[dna_mask_label==(cell_idx+1)]=0
            dna_mask_label[single_dna>0]=(cell_idx+1)
        else:
            # refine in each cell by removing small parts touching seperatinon bounary
            single_mem_bd = find_boundaries(cell_seg==(cell_idx+1), mode='inner')
            bd_idx = list(np.unique(single_dna_label[single_mem_bd>0]))
            if len(bd_idx)>0:
                for list_idx, dna_bd_idx in enumerate(bd_idx):
                    if np.count_nonzero(single_dna_label==dna_bd_idx)<600:
                        dna_mask_label[single_dna_label==dna_bd_idx]=0

    print('refinement is done.')
    print('checking for cell pairs ... ...')

    ################################################################
    ################# do cell pair detection #######################
    ################################################################
    
    pair_model = detect.import_model(
                    weight= "/allen/aics/assay-dev/users/Hyeonwoo/code/develop/trained_models/faster-rcnn/pair_detector/model_final.pth",
                    output = "/allen/aics/assay-dev/users/Jianxu",
                    config_file="/allen/aics/assay-dev/users/Hyeonwoo/code/develop/trained_models/faster-rcnn/pair_detector/aics_detection_train.yaml")

    print('detection model is loaded')
    rois_array = detect.predict(
                pair_model,
                filename=None, 
                image_array=img[[0],:,:,:].astype(np.float32), 
                channel=0, 
                shape="czyx", 
                normalization=True, 
                save_path=None,
                config_file="/allen/aics/assay-dev/users/Hyeonwoo/code/develop/trained_models/faster-rcnn/pair_detector/aics_detection_train.yaml")

    if len(rois_array)>0:
        # update variable name (the code was mostly copied from other scripts)
        mem_seg_whole = cell_seg
        nuc_seg_whole = dna_mask_label

        roi_list = rois_array[0]
        roi_score = rois_array[1].tolist()

        nuc_mip = np.amax(nuc_seg_whole, axis=0)
        cell_pairs = []
        cell_pairs_score = []
        cell_pairs_aux_score = []

        # step 1: go through all bounding boxes and extract possible pairs
        for roi2d_index, roi2d in enumerate(roi_list):
                
            nuc_crop = nuc_mip[roi2d[1]:roi2d[3], roi2d[0]:roi2d[2]]
            pair_candis = np.unique(nuc_crop[nuc_crop>0])
            this_score = roi_score[roi2d_index]

            # step 1.0: skip bounding box with extremely low score
            if this_score < pair_box_score_cutoff:
                continue
            
            # step 1.1: get all possile combinations
            pair_candi_valid = []
            for pair_i, cell_index in enumerate(pair_candis):
                single_cell = nuc_seg_whole==cell_index
                single_cell_mip = np.amax(single_cell, axis=0)
                single_cell_mip_crop = single_cell_mip[roi2d[1]:roi2d[3], roi2d[0]:roi2d[2]]
                # if one cell has over 60% out of the boundary box, the pair associated with this cell is not valid
                if np.count_nonzero(single_cell_mip_crop>0)/np.count_nonzero(single_cell_mip>0)>0.4:
                    pair_candi_valid.append(cell_index)

            # step 1.2: evaluate the validity of each combination
            cell_pairs_in_one_box = []
            cell_pairs_score_in_one_box = []
            cell_pairs_aux_score_in_one_box = []
            roi2d_size = (roi2d[3] - roi2d[1]) * (roi2d[2] - roi2d[0])
            for ii in np.arange(0,len(pair_candi_valid)-1):
                for jj in np.arange(ii+1,len(pair_candi_valid)):
                    if [pair_candi_valid[ii],pair_candi_valid[jj]] in cell_pairs:
                        continue

                    # step 1.2.1: for each pair, extract their mask insider the bounding box (denoted by B1)
                    #             and compute the bounding box of the cropped mask, denoted by B2
                    #             then, compare the size of B1 and B2. 
                    #             Because,we expect the bounding box returned by detector (i.e., B1)
                    #             should be a tight box around the true pair, the ratio of B2/B1 should not be 
                    #             too small. Otherwise, remove this potential combination
                    single_pair = np.logical_or(nuc_seg_whole==pair_candi_valid[jj], nuc_seg_whole==pair_candi_valid[ii]) 
                    single_pair_mip = np.amax(single_pair, axis=0)

                    single_pair_mip_crop = single_pair_mip[roi2d[1]:roi2d[3], roi2d[0]:roi2d[2]]
                    y_range = np.where(np.any(single_pair_mip_crop, axis=(0)))
                    x_range = np.where(np.any(single_pair_mip_crop, axis=(1)))
                    #print([pair_candi_valid[jj],pair_candi_valid[ii]])
                    inclusion_ratio = (y_range[0][-1] - y_range[0][0]) * (x_range[0][-1] - x_range[0][0]) / roi2d_size
                    #print(inclusion_ratio)
                    if inclusion_ratio < pair_inclusion_ratio_cutoff:
                        continue

                    #y_range_no_crop = np.where(np.any(single_pair_mip, axis=(0)))
                    #x_range_no_crop = np.where(np.any(single_pair_mip, axis=(1)))

                    # step 1.2.2: the two dna's of the pair should have comparable size
                    #             remove it if the size ratio is too small or too large
                    sz1=np.count_nonzero(nuc_seg_whole==pair_candi_valid[jj])
                    sz2=np.count_nonzero(nuc_seg_whole==pair_candi_valid[ii])
                    if sz1/sz2<1.5625 and sz1/sz2>0.64: 
                        cell_pairs_in_one_box.append([pair_candi_valid[ii],pair_candi_valid[jj]])
                        #cell_pairs_score_in_one_box.append(this_score)
                        cell_pairs_aux_score_in_one_box.append(inclusion_ratio)

            if len(cell_pairs_in_one_box)>0:
                # if there are more than one valid pair found in this box, we either keep the one 
                # with highest inclusion_ratio (if highest < 0.9) or remove pairs with inclusion_ratio
                # less than 0.9
                if np.max(cell_pairs_aux_score_in_one_box)<0.9:
                    best_pair_in_one_box = np.argmax(cell_pairs_aux_score_in_one_box)
                    cell_pairs.append(cell_pairs_in_one_box[best_pair_in_one_box])
                    cell_pairs_score.append(this_score)
                    cell_pairs_aux_score.append(cell_pairs_aux_score_in_one_box[best_pair_in_one_box])
                else:
                    for tmp_index, tmp_score in enumerate(cell_pairs_aux_score_in_one_box):
                        if tmp_score >= 0.9:
                            cell_pairs.append(cell_pairs_in_one_box[tmp_index])
                            cell_pairs_score.append(this_score)
                            cell_pairs_aux_score.append(tmp_score)
        
        if len(cell_pairs)>0:
            print('candidate cell pairs are found')

            # see one cell is associated with more than one pair
            if exist_double_assignment(cell_pairs):
                simple_pair, multi_pairs, multi_pair_index = find_multi_assignment(cell_pairs)
                multi_pairs_score = [cell_pairs_score[ii_tmp] for ii_tmp in multi_pair_index ]
                multi_pairs_aux_score = [cell_pairs_aux_score[ii_tmp] for ii_tmp in multi_pair_index ]
                while True:
                    current_best_pair = find_strongest_associate(multi_pairs_score, multi_pairs_aux_score)
                    idx_to_remove = prune_cell_pairs(multi_pairs, current_best_pair)
                    if not len(idx_to_remove)>0:
                        print('bug, during cell pair pruning')
                        if return_prediction:
                            return None, [dna_mask_pred, mem_pred, seed_pred]
                        else:
                            return None
                    multi_pairs = [ rm_v for rm_i, rm_v in enumerate(multi_pairs) if rm_i not in idx_to_remove]
                    multi_pairs_score = [ rm_v for rm_i, rm_v in enumerate(multi_pairs_score) if rm_i not in idx_to_remove]
                    multi_pairs_aux_score = [ rm_v for rm_i, rm_v in enumerate(multi_pairs_aux_score) if rm_i not in idx_to_remove]

                    if len(multi_pairs)==0 or (not exist_double_assignment(multi_pairs)):
                        cell_pairs = simple_pair + multi_pairs
                        break
                    else:
                        simple_pair_new, multi_pairs, multi_pair_index_new = find_multi_assignment(multi_pairs)
                        simple_pair = simple_pair + simple_pair_new
                        multi_pairs_score =  [multi_pairs_score[ii_tmp] for ii_tmp in multi_pair_index_new]
                        multi_pairs_aux_score = [multi_pairs_aux_score[ii_tmp] for ii_tmp in multi_pair_index_new]   
            for index, pair_ids in enumerate(cell_pairs):
                nuc_seg = np.logical_or(nuc_seg_whole == pair_ids[0], nuc_seg_whole == pair_ids[1])
                mem_seg = np.logical_or(mem_seg_whole == pair_ids[0], mem_seg_whole == pair_ids[1])  
                mem_seg = binary_closing(mem_seg, selem=ball(3))
                
                mem_seg_whole[mem_seg>0] = pair_ids[0]
                nuc_seg_whole[nuc_seg>0] = pair_ids[0]


            #### put back to the original variable
            cell_seg = mem_seg_whole
            dna_mask_label = nuc_seg_whole

    # re-squence the index
    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))

    # propagate the cell index to dna
    dna_mask_label[cell_seg==0]=0
    dna_mask_label[dna_mask_label>0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    # create contours
    seg_mem_contour = np.zeros_like(cell_seg)
    seg_dna_contour = np.zeros_like(dna_mask_label)
    valid_cell_index = np.unique(cell_seg[cell_seg>0])
    for index, cid in enumerate(valid_cell_index): 
        single_mem = cell_seg==cid
        single_dna = dna_mask_label==cid
        single_mem_contour = np.zeros_like(single_mem)
        single_dna_contour = np.zeros_like(single_dna)
        for zz in range(single_mem.shape[0]):
            if np.any(single_mem[zz,:,:]):
                single_mem_contour[zz,:,:] = find_boundaries(single_mem[zz, :, :] > 0, mode='inner')
            if np.any(single_dna[zz,:,:]):
                single_dna_contour[zz,:,:] = find_boundaries(single_dna[zz, :, :] > 0, mode='inner')

        seg_mem_contour[single_mem_contour>0] = cid
        seg_dna_contour[single_dna_contour>0] = cid
    '''


    combined_seg = np.stack([dna_mask_label.astype(np.uint8), \
            cell_seg.astype(np.uint8)], axis=1)
    return combined_seg

    '''
    ################################################################
    # remove all border-touching cells
    ################################################################

    bd_idx = list(np.unique(cell_seg[boundary_mask>0]))
    cell_seg_bd = np.zeros_like(cell_seg)
    for cid in bd_idx:
        if cid>0:
            cell_seg_bd[cell_seg==cid] = cid
            cell_seg[cell_seg==cid]=0

    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))
    num_valid_cell = cell_seg.max()
    cell_seg_bd, _tmp , _tmp2 = relabel_sequential(cell_seg_bd.astype(np.uint8))
    cell_seg_bd = cell_seg_bd + num_valid_cell
    cell_seg_bd[cell_seg_bd==num_valid_cell]=0

    # propagate the cell index to dna
    dna_mask_label_bd = dna_mask_label.copy()
    dna_mask_label[cell_seg==0]=0
    dna_mask_label[dna_mask_label>0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    dna_mask_label_bd[cell_seg_bd==0]=0
    dna_mask_label_bd[dna_mask_label_bd>0] = 1
    dna_mask_label_bd = dna_mask_label_bd * cell_seg_bd

    # create contours
    seg_mem_contour = np.zeros_like(dna_mask_label)
    seg_dna_contour = np.zeros_like(dna_mask_label)
    valid_cell_index = np.unique(dna_mask_label[dna_mask_label>0])
    for index, cid in enumerate(valid_cell_index): 
        single_mem = cell_seg==cid
        single_dna = dna_mask_label==cid
        single_mem_contour = np.zeros_like(single_mem)
        single_dna_contour = np.zeros_like(single_dna)
        for zz in range(single_mem.shape[0]):
            if np.any(single_mem[zz,:,:]):
                single_mem_contour[zz,:,:] = find_boundaries(single_mem[zz, :, :] > 0, mode='inner')
            if np.any(single_dna[zz,:,:]):
                single_dna_contour[zz,:,:] = find_boundaries(single_dna[zz, :, :] > 0, mode='inner')

        seg_mem_contour[single_mem_contour>0] = cid
        seg_dna_contour[single_dna_contour>0] = cid

    seg_mem_contour_bd = np.zeros_like(dna_mask_label)
    seg_dna_contour_bd = np.zeros_like(dna_mask_label)
    bd_cell_index = np.unique(dna_mask_label_bd[dna_mask_label_bd>0])
    for index, cid in enumerate(bd_cell_index): 
        single_mem = cell_seg_bd==cid
        single_dna = dna_mask_label_bd==cid
        single_mem_contour = np.zeros_like(single_mem)
        single_dna_contour = np.zeros_like(single_dna)
        for zz in range(single_mem.shape[0]):
            if np.any(single_mem[zz,:,:]):
                single_mem_contour[zz,:,:] = find_boundaries(single_mem[zz, :, :] > 0, mode='inner')
            if np.any(single_dna[zz,:,:]):
                single_dna_contour[zz,:,:] = find_boundaries(single_dna[zz, :, :] > 0, mode='inner')

        seg_mem_contour_bd[single_mem_contour>0] = cid
        seg_dna_contour_bd[single_dna_contour>0] = cid

    combined_seg = np.stack([dna_mask_label.astype(np.uint8), dna_mask_label_bd.astype(np.uint8), \
            cell_seg.astype(np.uint8), cell_seg_bd.astype(np.uint8), seg_dna_contour.astype(np.uint8), \
            seg_dna_contour_bd.astype(np.uint8), seg_mem_contour.astype(np.uint8), \
            seg_mem_contour_bd.astype(np.uint8)], axis=1)
    return combined_seg
    #if return_prediction:
    #    return cell_seg, dna_mask_label, [dna_mask_pred, mem_pred, seed_pred]
    #else:
    #    return cell_seg, dna_mask_label
    '''
