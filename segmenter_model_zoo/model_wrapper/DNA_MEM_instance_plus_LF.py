import os
import numpy as np
from aicsimageio import AICSImage
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import ball, dilation, disk, binary_closing
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.segmentation import relabel_sequential, find_boundaries 
from tifffile import imsave
from aicsmlsegment.utils import background_sub, simple_norm
import itk
from collections import Counter

from cell_detector import detect
from segmenter_model_zoo.quilt_utils import QuiltModelZoo

mem_pre_cut_th = 0.2 + 0.25  # 0.2  # 0.02
seed_bw_th = 0.75  # 0.90
dna_mask_bw_th = 0.5  # 0.7 
min_seed_size = 6000  # 9000 # 3800

pair_box_score_cutoff = 0.75
pair_inclusion_ratio_cutoff = 0.65

mem_bf_cut = 0.25
dna_bf_cutoff = 1.5

flat_se = np.zeros((5, 5, 5), dtype=np.uint8)
flat_se[2, :, :] = 1


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def find_strongest_associate(main_score, aux_score):
    score = []
    for ii in range(len(main_score)):
        score.append((main_score[ii], aux_score[ii]))
    score_array = np.array(score, dtype='<f4,<f4')
    weight_order = score_array.argsort()
    return weight_order.argmax()


def exist_double_assignment(cell_pairs):
    nodes = set(x for ll in cell_pairs for x in ll)
    num_node = len(nodes)
    num_pair = len(cell_pairs)

    return not num_pair * 2 == num_node


def find_multi_assignment(cell_pairs):
    nodes = [x for ll in cell_pairs for x in ll]
    node_count = Counter(nodes)
    simple_pair = []
    multi_pair = []
    multi_pair_index = []
    for idx, p in enumerate(cell_pairs):
        if node_count[p[0]] > 1 or node_count[p[1]] > 1:
            multi_pair.append(p)
            multi_pair_index.append(idx)
        else:
            simple_pair.append(p)

    return simple_pair, multi_pair, multi_pair_index


def prune_cell_pairs(multi_pair, current_best_pair):

    p_best = multi_pair[current_best_pair]
    idx_to_remove = []
    for idx, p in enumerate(multi_pair):
        if idx == current_best_pair:
            continue
        if p[0] in p_best or p[1] in p_best:
            idx_to_remove.append(idx)

    return idx_to_remove


def getLargestCC(labels, is_label=True):

    if is_label:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        sub_labels = label(labels > 0, connectivity=3, return_num=False)
        largestCC = sub_labels == np.argmax(np.bincount(sub_labels.flat)[1:]) + 1

    return largestCC


def SegModule(
    img=None,
    model_list=None,
    prune_border=False,
    filename=None,
    index=None,
    return_prediction=False,
    two_camera=False
):

    # if two_camera:
    #    mem_bf_cut = 1.9 ### only use lf pred for membrane top and bottom 
    # (the seperation in two camera could be wrong)
    #    dna_bf_cutoff = 1.45 ### decrease cutoff after turn tta off
    # else:
    #    mem_bf_cut = 0.25
    #    dna_bf_cutoff = 1.5
    # model order: dna_mask, cellmask, dna_seed

    if img is None:
        # load the image 
        reader = AICSImage(filename)
        img = reader.data[0, index, :, :, :]

    # make sure the image has 4 dimensions
    # assert len(img.shape)==4 and img.shape[0]==3
    if not (len(img.shape) == 4 and img.shape[0] == 3 and img.shape[1] >= 32):
        print('bad data, dimension crashed')
        if return_prediction:
            return None, None
        else:
            return None

    ###########################################################
    # part 1: prepare data
    ###########################################################

    # input channel order:
    # first = dna; second = cell mask; third = bf

    # extract dna channel 
    dna_img = img[0, :, :, :].copy()
    dna_img[dna_img > 60000] = dna_img.min()
    dna_img = background_sub(dna_img, 50)
    dna_img = simple_norm(dna_img, 2.5, 10)
    imsave('dna_norm.tiff', dna_img)

    # extra cellmask channel
    mem_img = img[1, :, :, :].copy()
    mem_img[mem_img > 60000] = mem_img.min()
    mem_img = background_sub(mem_img, 50)
    mem_img = simple_norm(mem_img, 2, 11)
    imsave('cell_norm.tiff', mem_img)

    print('image normalization is done')
    print('applying all DL models ... ...')

    ###########################################################
    # part 2: run predictions
    ###########################################################

    # model 1: dna_mask
    dna_mask_pred = model_list[0].apply_on_single_zstack(
        dna_img,
        already_normalized=True,
        cutoff=-1
    )
    dna_mask_bw = dna_mask_pred > dna_mask_bw_th
    imsave('pred_dna.tiff', dna_mask_pred)

    # model 2: cell edge
    mem_pred = model_list[1].apply_on_single_zstack(
        mem_img,
        already_normalized=True,
        cutoff=-1
    )
    imsave('pred_cell.tiff', mem_pred)
    # model 3: dna_seed
    seed_pred = model_list[2].apply_on_single_zstack(
        dna_img,
        already_normalized=True,
        cutoff=-1
    )
    seed_bw = seed_pred > seed_bw_th
    imsave('pred_seed.tiff', seed_pred)

    # model 4: dna from bf
    if two_camera:
        dna_bf_pred = model_list[3].apply_on_single_zstack(
            input_img=img[2, :, :, :], 
            use_tta=False
        )
    else:
        dna_bf_pred = model_list[3].apply_on_single_zstack(
            input_img=img[2, :, :, :],
            use_tta=True
        )
    dna_bf_bw = dna_bf_pred > dna_bf_cutoff
    imsave('lf_dna.tiff', dna_mask_pred)

    # model 5: mem from bf
    if two_camera:
        mem_bf_pred = model_list[4].apply_on_single_zstack(
            input_img=img[2, :, :, :],
            use_tta=False
        )
    else:
        mem_bf_pred = model_list[4].apply_on_single_zstack(
            input_img=img[2, :, :, :],
            use_tta=True
        )
    imsave('lf_mem.tiff', mem_bf_pred)
    print('predictions are done.')

    ###########################################################
    # part 3: merge bf based prediction into dye based prediction
    ###########################################################

    # adjust mem_pred by bf
    mem_bf_trust = np.zeros_like(mem_pred)
    mem_bf_trust[mem_bf_pred > mem_bf_cut] = 1
    mem_pred = mem_pred + mem_bf_trust * 0.1

    mem_bf_trust_1 = dilation(mem_bf_trust > 0, selem=flat_se)
    mem_bf_trust_1 = mem_bf_trust_1.astype(np.uint8)
    mem_bf_trust_1[mem_bf_trust_1 > 0] = 1
    mem_pred = mem_pred + mem_bf_trust_1 * 0.1

    mem_bf_trust_2 = dilation(mem_bf_trust_1 > 0, selem=flat_se)
    mem_bf_trust_2 = mem_bf_trust_2.astype(np.uint8)
    mem_bf_trust_2[mem_bf_trust_2 > 0] = 1
    mem_pred = mem_pred + mem_bf_trust_2 * 0.1

    mem_bf_trust_3 = dilation(mem_bf_trust_2 > 0, selem=flat_se)
    mem_bf_trust_3 = mem_bf_trust_3.astype(np.uint8)
    mem_bf_trust_3[mem_bf_trust_3 > 0] = 1
    mem_pred = mem_pred + mem_bf_trust_3 * 0.1

    # prepare separation boundary
    tmp_mem = mem_pred > mem_pre_cut_th
    for zz in range(tmp_mem.shape[0]):
        if np.any(tmp_mem[zz, :, :]):
            tmp_mem[zz, :, :] = dilation(tmp_mem[zz, :, :], selem=disk(1))

    # cut seed 
    seed_bw[tmp_mem > 0] = 0

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
    boundary_mask[:, -4:, :] = 1
    boundary_mask[:, :, :4] = 1
    boundary_mask[:, :, -4:] = 1

    bd_seed_on_hold = np.zeros_like(seed_bw)
    bd_idx = list(np.unique(seed_label[boundary_mask > 0]))
    for index, cid in enumerate(bd_idx):
        if cid > 0:
            bd_seed_on_hold[seed_label == cid] = 1

    seed_bw = remove_small_objects(seed_bw, min_size=min_seed_size, connectivity=1)

    # finalize seed (add back the seeds on hold)
    seed_bw[bd_seed_on_hold > 0] = 1

    ##################################################
    # adjust seed and dna mask
    ##################################################

    # cut the prediction
    dna_bf_bw[tmp_mem > 0] = 0

    # merge in seed_before_pruning
    # the reason is sometimes for late M7 dna, it is small (due to eroded seed) 
    # and could be falsely pruned, and the labelfree prediction on mitotic dna 
    # is not always reliable. so, we merge the seed_before_pruning and the 
    # dna_bf_bw (to pass the size filter) and make the rescue step more effective
    dna_bf_bw_augmented = np.logical_or(dna_bf_bw, seed_bw_before_pruning)

    # apply different min size for touching/not_touching border
    dna_bf_bw_augmented = remove_small_objects(dna_bf_bw_augmented, min_size=20)
    dna_bf_bwlab = label(dna_bf_bw_augmented , connectivity=1)

    # save the objs that are touching boundary
    bd_bf_bw_on_hold = np.zeros_like(dna_bf_bw_augmented)
    bd_bf_idx = list(np.unique(dna_bf_bwlab[boundary_mask > 0]))
    for index, cid in enumerate(bd_bf_idx):
        if cid > 0:
            bd_bf_bw_on_hold[dna_bf_bwlab == cid] = 1

    dna_bf_bw_augmented = remove_small_objects(
        dna_bf_bw_augmented,
        min_size=min_seed_size,
        connectivity=1
    )

    # finalize bf bw (add back the objs on hold)
    dna_bf_bw_augmented[bd_bf_bw_on_hold > 0] = 1
    dna_bf_label, num_pred_dna = label(dna_bf_bw_augmented , return_num=True)

    # merge into seed and dna-mask
    for ii in range(num_pred_dna):
        # if one obj does not overlap with any seed
        single_extra = dna_bf_label == (ii + 1)
        if np.count_nonzero(np.logical_and(single_extra, seed_bw)) < 50:
            seed_bw[single_extra > 0] = 1
            # if this seed is missing, assumes the signle of this dna is problematic
            dna_mask_bw[single_extra > 0] = 1 
        elif np.count_nonzero(np.logical_and(single_extra, dna_mask_bw)) < 50:
            dna_mask_bw[single_extra > 0] = 1

    ###########################################################
    # part 4: prepare for watershed image
    ###########################################################

    # find the stack bottom
    stack_bottom = 0
    for zz in np.arange(3, tmp_mem.shape[0] // 2):
        if np.count_nonzero(tmp_mem[zz, :, :] > 0) > 0.5 * \
           tmp_mem.shape[1] * tmp_mem.shape[2]:
            stack_bottom = zz
            break

    # find the stack top
    stack_top = mem_pred.shape[0] - 1
    for zz in np.arange(mem_pred.shape[0] - 1, mem_pred.shape[0] // 2 + 1, -1):
        if np.count_nonzero(tmp_mem[zz, :, :] > 0) > 64:
            stack_top = zz 
            break

    # prune mem_pred
    if stack_bottom == 0:
        mem_pred[0, :, :] = 0.0000001
    else:
        mem_pred[:stack_bottom, :, :] = 0.0000001
    mem_pred[stack_top:, :, :] = 0.0000001

    #############################################################
    # part 5: prepare for watershed seed
    #############################################################
    seed_label, seed_num = label(seed_bw, return_num=True, connectivity=1)
    if stack_bottom == 0:
        seed_label[0, :, :] = seed_num + 1
    else:
        seed_label[:stack_bottom, :, :] = seed_num + 1
    seed_label[stack_top:, :, :] = seed_num + 2

    ################################################################
    # part 6: get cell instance segmentation
    ################################################################
    # cell_seg = watershed(mem_pred, seed_label, watershed_line=True)
    raw0 = mem_pred.astype(np.float32)
    raw_itk = itk.GetImageFromArray(raw0)
    seed_itk = itk.GetImageFromArray(seed_label.astype(np.int16))
    seg_itk = itk.morphological_watershed_from_markers_image_filter(
        raw_itk,
        marker_image=seed_itk,
        fully_connected=True,
        mark_watershed_line=False
    )
    cell_seg = itk.GetArrayFromImage(seg_itk)

    cell_seg[cell_seg == seed_num + 1] = 0
    cell_seg[cell_seg == seed_num + 2] = 0

    print('watershed based cell segmentation is done.')

    ################################################################
    # part 7: refine cell segmentation near bottom
    ################################################################
    # estimate colony coverage size
    colony_coverage = np.amax(tmp_mem.astype(np.uint8), axis=0)
    colony_coverage_size = np.count_nonzero(colony_coverage.flat > 0)

    step_down_z = stack_bottom - 1
    for zz in np.arange(stack_bottom - 1, cell_seg.shape[0] // 2):
        if np.count_nonzero(cell_seg[zz, :, :] > 0) > 0.8 * colony_coverage_size:
            step_down_z = zz
            break 

    for zz in np.arange(stack_bottom, step_down_z, 1):
        cell_seg[zz, :, :] = cell_seg[step_down_z, :, :]

    print('stack bottom has been properly updated.')

    # remove small cells due to failure / noise
    for ii in np.unique(cell_seg[cell_seg > 0]):
        this_one_cell = cell_seg == ii
        this_dna = dna_mask_bw.copy()
        this_dna[this_one_cell == 0] = 0
        # small "cell" or "dna"
        if np.count_nonzero(this_one_cell > 0) < 70000 or \
           np.count_nonzero(this_dna > 0) < 1000 or \
           np.count_nonzero(getLargestCC(this_dna, is_label=False)) < 10000: 
            cell_seg[this_one_cell > 0] = 0

    # false clip check (mem channel)
    # i.e. the segmentation should not appear in first of last z-slice
    z_range_mem = np.where(np.any(cell_seg, axis=(1, 2)))
    z_range_mem = z_range_mem[0]
    if len(z_range_mem) == 0 or z_range_mem[0] == 0 or \
       z_range_mem[-1] == cell_seg.shape[0] - 1:
        print('exit because false clip or bad floaty is detected in mem channel')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    # relabel the index in case altered when dumping the bottom
    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))

    print('size based QC is done')

    ################################################################
    # get dna instance segmentation
    ################################################################
    # make sure dna is not out of membrane
    dna_mask_bw[cell_seg == 0] = 0

    # propagate the cell index to dna
    dna_mask_label = np.zeros_like(cell_seg)
    dna_mask_label[dna_mask_bw > 0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    ###################################
    # false clip check (dna channel)
    # i.e., the segmentation should not appear in first of last z-slice
    z_range_dna = np.where(np.any(dna_mask_label, axis=(1, 2)))
    z_range_dna = z_range_dna[0]
    if len(z_range_dna) == 0 or z_range_dna[0] == 0 or \
       z_range_dna[-1] == dna_mask_label.shape[0] - 1:
        print('exit because false clip or bad floaty is detected in dna channel')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    if dna_mask_label.max() < 3:  # if only a few cells left, just throw it away
        print('exit because only very few cells are segmented, maybe a bad image')
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    print('refining dna masks ... ...')

    # get the index touching border
    bd_idx = list(np.unique(cell_seg[boundary_mask > 0]))

    # refine dna 
    num_cell = cell_seg.max()
    for cell_idx in range(num_cell):
        if (cell_idx + 1) in bd_idx:
            # no need to refine, because this will be ignored in the real analysis
            continue

        # empty dna should have been removed
        if not np.any(dna_mask_label == (cell_idx + 1)):  
            print('bug, empty dna is found, but should not')
            if return_prediction:
                return None, [dna_mask_pred, mem_pred, seed_pred]
            else:
                return None

        # extract the largest component from the dna segmentation within this cell
        single_dna = dna_mask_label == (cell_idx + 1)
        single_dna_label, num_obj = label(single_dna, return_num=True, connectivity=3)
        if num_obj == 1:
            largest_label = single_dna_label > 0
        elif num_obj > 1:
            largest_label = getLargestCC(single_dna_label)
        else:
            print('bug occurs in processing the pair ... ')
            if return_prediction:
                return None, [dna_mask_pred, mem_pred, seed_pred]
            else:
                return None

        # ######################################################################
        # choose different pruning method based on morphology
        #   - option 1: not melt (interphase or early/late mitosis)
        #               keep the largest connected component after 
        #               filling holes and removing small objects
        #   - option 2: if melted (mitosis)
        #               only clean up small objects near the cutting boundary
        #               with neighbor cells (very like a not-precise cut)
        # to decide which one to use, we compare the ratio between the dna mask 
        # and dna seed within this cell, because dna seed will be much smaller 
        # than dna mask if a mitotic cell has not started to "melt"
        # ######################################################################
        mask_size_in_this_cell = np.count_nonzero(largest_label)
        seed_size_in_this_cell = np.count_nonzero(seed_label == (cell_idx + 1))
        ratio_check = seed_size_in_this_cell / mask_size_in_this_cell
        if ratio_check < 0.75:  # interphase
            # prune dna mask
            for zz in range(single_dna.shape[0]):
                if np.any(single_dna[zz, :, :]):
                    single_dna[zz, :, :] = binary_fill_holes(single_dna[zz, :, :])
                    single_dna[zz, :, :] = remove_small_objects(single_dna[zz, :, :],
                                                                min_size=100)
            single_dna = remove_small_objects(single_dna, min_size=2500)
            dna_mask_label[dna_mask_label == (cell_idx + 1)] = 0
            dna_mask_label[single_dna > 0] = (cell_idx + 1)
        else:
            # refine in each cell by removing small parts touching seperatinon bounary
            single_mem_bd = find_boundaries(cell_seg == (cell_idx + 1), mode='inner')
            bd_idx = list(np.unique(single_dna_label[single_mem_bd > 0]))
            if len(bd_idx) > 0:
                for list_idx, dna_bd_idx in enumerate(bd_idx):
                    if np.count_nonzero(single_dna_label == dna_bd_idx) < 600:
                        dna_mask_label[single_dna_label == dna_bd_idx] = 0

    print('refinement is done.')
    print('checking for cell pairs ... ...')

    # ###############################################################
    # ################ do cell pair detection #######################
    # ###############################################################

    # first, load the trained object detection model
    _tmp_dir = str(model_list[-1])
    _local_tmp = _tmp_dir + os.sep + '_tmp_cell_pair_detection_model.pth'
    _local_config_tmp = _tmp_dir + os.sep + '_tmp_cell_pair_detection_config.yaml'
    if not os.path.exists(_local_tmp):
        model_helper = QuiltModelZoo()
        model_helper.download_model("cell_pair_det_prod", _local_tmp)
        model_helper.download_model("cell_pair_det_prod_config", _local_config_tmp)

    # load the model
    pair_model = detect.import_model(
        weight=_local_tmp,
        output=_tmp_dir,
        config_file=_local_config_tmp
    )
    print('detection model is loaded')

    # run prediction
    rois_array = detect.predict(
        pair_model,
        filename=None, 
        image_array=img[[0], :, :, :].astype(np.float32), 
        channel=0, 
        shape="czyx", 
        normalization=True, 
        save_path=None,
        config_file=_local_config_tmp
    )

    # check if any potential pairs
    if len(rois_array) > 0:

        # extract the coordinates and confidence score of each box
        roi_list = rois_array[0]
        roi_score = rois_array[1].tolist()

        # get nuc labels in max-proj
        nuc_mip = np.amax(dna_mask_label, axis=0)

        # step 1: go through all bounding boxes and extract possible pairs
        cell_pairs = []
        cell_pairs_score = []
        cell_pairs_aux_score = []
        for roi2d_index, roi2d in enumerate(roi_list): 

            # crop out the nuc seg within the box and get ID of candidate cells
            # there may be more than 2 IDs
            nuc_crop = nuc_mip[roi2d[1]:roi2d[3], roi2d[0]:roi2d[2]]
            pair_candis = np.unique(nuc_crop[nuc_crop > 0])

            # get confidence score of this box
            this_score = roi_score[roi2d_index]

            # step 1.0: skip bounding box with extremely low score
            if this_score < pair_box_score_cutoff:
                continue

            # step 1.1: get all possile combinations
            pair_candi_valid = []
            for pair_i, cell_index in enumerate(pair_candis):
                single_cell = dna_mask_label == cell_index
                single_cell_mip = np.amax(single_cell, axis=0)
                single_cell_mip_crop = single_cell_mip[roi2d[1]: roi2d[3],
                                                       roi2d[0]: roi2d[2]]
                # if one cell has over 60% out of the boundary box, the pair 
                # associated with this cell is not valid, otherwise keep it
                # as candidates
                area_within_box = np.count_nonzero(single_cell_mip_crop > 0)
                area_of_whole_cell = np.count_nonzero(single_cell_mip > 0)
                if area_within_box / area_of_whole_cell > 0.4:
                    pair_candi_valid.append(cell_index)

            # step 1.2: evaluate the validity of each candidate combination
            cell_pairs_in_one_box = []
            cell_pairs_aux_score_in_one_box = []
            roi2d_size = (roi2d[3] - roi2d[1]) * (roi2d[2] - roi2d[0])
            for ii in np.arange(0, len(pair_candi_valid) - 1):
                for jj in np.arange(ii + 1, len(pair_candi_valid)):
                    if [pair_candi_valid[ii], pair_candi_valid[jj]] in cell_pairs:
                        continue

                    # step 1.2.1: 
                    # for each pair, extract their mask inside the box (denoted by B1)
                    # and compute the bounding box of the cropped mask, denoted by B2
                    # then, compare the size of B1 and B2. 
                    # Because, we expect the bounding box returned by detector (ie, B1)
                    # should be a tight box around the true pair, the ratio of B2/B1 
                    # should not be too small. Otherwise, remove this potential pair
                    single_pair = np.logical_or(
                        dna_mask_label == pair_candi_valid[jj],
                        dna_mask_label == pair_candi_valid[ii]
                    ) 
                    single_pair_mip = np.amax(single_pair, axis=0)

                    single_pair_mip_crop = single_pair_mip[roi2d[1]:roi2d[3],
                                                           roi2d[0]:roi2d[2]]
                    y_range = np.where(np.any(single_pair_mip_crop, axis=(0)))
                    x_range = np.where(np.any(single_pair_mip_crop, axis=(1)))
                    box_B2 = (y_range[0][-1] - y_range[0][0]) * \
                             (x_range[0][-1] - x_range[0][0])
                    inclusion_ratio = box_B2 / roi2d_size
                    if inclusion_ratio < pair_inclusion_ratio_cutoff:
                        continue

                    # step 1.2.2: the two dna's of the pair should have comparable size
                    #             remove it if the size ratio is too small or too large
                    sz1 = np.count_nonzero(dna_mask_label == pair_candi_valid[jj])
                    sz2 = np.count_nonzero(dna_mask_label == pair_candi_valid[ii])
                    if sz1 / sz2 < 1.5625 and sz1 / sz2 > 0.64: 
                        cell_pairs_in_one_box.append(
                            [pair_candi_valid[ii], pair_candi_valid[jj]]
                        )
                        cell_pairs_aux_score_in_one_box.append(inclusion_ratio)

            if len(cell_pairs_in_one_box) > 0:
                # if there are more than one valid pair found in this box, we either 
                # keep the one with highest inclusion_ratio (if highest < 0.9) or 
                # remove pairs with inclusion_ratio less than 0.9
                if np.max(cell_pairs_aux_score_in_one_box) < 0.9:
                    best_pair_in_one_box = np.argmax(cell_pairs_aux_score_in_one_box)
                    cell_pairs.append(cell_pairs_in_one_box[best_pair_in_one_box])
                    cell_pairs_score.append(this_score)
                    cell_pairs_aux_score.append(
                        cell_pairs_aux_score_in_one_box[best_pair_in_one_box]
                    )
                else:
                    for tmp_index, tmp_score in enumerate(
                        cell_pairs_aux_score_in_one_box
                    ):
                        if tmp_score >= 0.9:
                            cell_pairs.append(cell_pairs_in_one_box[tmp_index])
                            cell_pairs_score.append(this_score)
                            cell_pairs_aux_score.append(tmp_score)

        if len(cell_pairs) > 0:
            print('candidate cell pairs are found')

            # checking double assignment
            # see if one cell is associated with more than one pair
            # this is possible, for example where two mitosis happen next to
            # each other
            if exist_double_assignment(cell_pairs):
                # seperate out which are multi-assignment
                simple_pair, multi_pairs, multi_pair_index = \
                    find_multi_assignment(cell_pairs)
                # for each multi-assignment, evaluate each of them
                multi_pairs_score = [
                    cell_pairs_score[ii_tmp] for ii_tmp in multi_pair_index
                ]
                multi_pairs_aux_score = [
                    cell_pairs_aux_score[ii_tmp] for ii_tmp in multi_pair_index
                ]

                # prune, until no more multi-assignment
                while True:
                    current_best_pair = find_strongest_associate(multi_pairs_score,
                                                                 multi_pairs_aux_score)
                    idx_to_remove = prune_cell_pairs(multi_pairs, current_best_pair)
                    if not len(idx_to_remove) > 0:
                        print('bug, during cell pair pruning')
                        if return_prediction:
                            return None, [dna_mask_pred, mem_pred, seed_pred]
                        else:
                            return None
                    multi_pairs = [rm_v for rm_i, rm_v in enumerate(multi_pairs) if rm_i not in idx_to_remove]  # noqa: E501
                    multi_pairs_score = [rm_v for rm_i, rm_v in enumerate(multi_pairs_score) if rm_i not in idx_to_remove]  # noqa: E501
                    multi_pairs_aux_score = [rm_v for rm_i, rm_v in enumerate(multi_pairs_aux_score) if rm_i not in idx_to_remove]  # noqa: E501

                    if len(multi_pairs) == 0 or \
                       (not exist_double_assignment(multi_pairs)):
                        cell_pairs = simple_pair + multi_pairs
                        break
                    else:
                        simple_pair_new, multi_pairs, multi_pair_index_new = \
                            find_multi_assignment(multi_pairs)
                        simple_pair = simple_pair + simple_pair_new
                        multi_pairs_score = [multi_pairs_score[ii_tmp] for ii_tmp in multi_pair_index_new]  # noqa: E501
                        multi_pairs_aux_score = [multi_pairs_aux_score[ii_tmp] for ii_tmp in multi_pair_index_new]   # noqa: E501

            # finally, ok to fix the indices
            for index, pair_ids in enumerate(cell_pairs):
                nuc_seg = np.logical_or(dna_mask_label == pair_ids[0],
                                        dna_mask_label == pair_ids[1])
                mem_seg = np.logical_or(cell_seg == pair_ids[0],
                                        cell_seg == pair_ids[1])  
                mem_seg = binary_closing(mem_seg, selem=ball(3))

                cell_seg[mem_seg > 0] = pair_ids[0]
                dna_mask_label[nuc_seg > 0] = pair_ids[0]

    # re-squence the index
    cell_seg, _tmp , _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))

    # propagate the cell index to dna
    dna_mask_label[cell_seg == 0] = 0
    dna_mask_label[dna_mask_label > 0] = 1
    dna_mask_label = dna_mask_label * cell_seg

    # create contours
    seg_mem_contour = np.zeros_like(cell_seg)
    seg_dna_contour = np.zeros_like(dna_mask_label)
    valid_cell_index = np.unique(cell_seg[cell_seg > 0])
    for index, cid in enumerate(valid_cell_index): 
        single_mem = cell_seg == cid
        single_dna = dna_mask_label == cid
        single_mem_contour = np.zeros_like(single_mem)
        single_dna_contour = np.zeros_like(single_dna)
        for zz in range(single_mem.shape[0]):
            if np.any(single_mem[zz, :, :]):
                single_mem_contour[zz, :, :] = \
                    find_boundaries(single_mem[zz, :, :] > 0, mode='inner')
            if np.any(single_dna[zz, :, :]):
                single_dna_contour[zz, :, :] = \
                    find_boundaries(single_dna[zz, :, :] > 0, mode='inner')

        seg_mem_contour[single_mem_contour > 0] = cid
        seg_dna_contour[single_dna_contour > 0] = cid

    combined_seg = np.stack(
        [
            dna_mask_label.astype(np.uint8),
            cell_seg.astype(np.uint8), seg_dna_contour.astype(np.uint8),
            seg_mem_contour.astype(np.uint8)
        ], axis=1
    )
    return combined_seg
