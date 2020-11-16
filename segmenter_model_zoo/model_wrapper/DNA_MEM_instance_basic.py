import numpy as np
from aicsimageio import AICSImage
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, disk
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.segmentation import relabel_sequential, find_boundaries
from aicsmlsegment.utils import background_sub, simple_norm
import itk

from segmenter_model_zoo.utils import getLargestCC

mem_pre_cut_th = 0.2  # 0.02
seed_bw_th = 0.90
dna_mask_bw_th = 0.5  # 0.7
min_seed_size = 3800  # 9000 # 3800


def SegModule(
    img=None,
    model_list=None,
    prune_border=False,
    filename=None,
    index=None,
    return_prediction=False,
    two_camera=False,
    output_type="default",
):
    # model order: dna_mask, cellmask, dna_seed
    if img is None:
        # load the image
        reader = AICSImage(filename)
        img = reader.data[0, index, :, :, :]
    # make sure the image has 4 dimensions
    if not (len(img.shape) == 4 and img.shape[0] == 2):
        print("bad data, dimension crashed")
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
    # imsave('dna_norm.tiff', dna_img)

    # extra cellmask channel
    mem_img = img[1, :, :, :].copy()
    mem_img[mem_img > 60000] = mem_img.min()
    mem_img = background_sub(mem_img, 50)
    mem_img = simple_norm(mem_img, 2, 11)
    # imsave('cell_norm.tiff', mem_img)

    print("image normalization is done")
    print("applying all DL models ... ...")

    ###########################################################
    # part 2: run predictions
    ###########################################################

    # model 1: dna_mask
    dna_mask_pred = model_list[0].apply_on_single_zstack(
        dna_img, already_normalized=True, cutoff=-1
    )
    dna_mask_bw = dna_mask_pred > dna_mask_bw_th
    # imsave('pred_dna.tiff', dna_mask_pred)

    # model 2: cell edge
    mem_pred = model_list[1].apply_on_single_zstack(
        mem_img, already_normalized=True, cutoff=-1
    )
    # imsave('pred_cell.tiff', mem_pred)

    # model 3: dna_seed
    seed_pred = model_list[2].apply_on_single_zstack(
        dna_img, already_normalized=True, cutoff=-1
    )
    seed_bw = seed_pred > seed_bw_th
    # imsave('pred_seed.tiff', seed_pred)
    print("predictions are done.")

    #############################################################
    # part 3: prepare seed
    #############################################################
    # prepare separation boundary first
    tmp_mem = mem_pred > mem_pre_cut_th
    for zz in range(tmp_mem.shape[0]):
        if np.any(tmp_mem[zz, :, :]):
            tmp_mem[zz, :, :] = dilation(tmp_mem[zz, :, :], selem=disk(1))

    # cut seed
    seed_bw[tmp_mem > 0] = 0

    # prune the seed
    seed_bw = remove_small_objects(seed_bw, min_size=20)
    seed_label = label(seed_bw, connectivity=1)

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

    ###########################################################
    # part 4: prepare for watershed image
    ###########################################################

    # find the stack bottom
    stack_bottom = 0
    for zz in np.arange(3, tmp_mem.shape[0] // 2):
        if (
            np.count_nonzero(tmp_mem[zz, :, :] > 0)
            > 0.5 * tmp_mem.shape[1] * tmp_mem.shape[2]
        ):
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
        raw_itk, marker_image=seed_itk, fully_connected=True, mark_watershed_line=False
    )
    cell_seg = itk.GetArrayFromImage(seg_itk)

    cell_seg[cell_seg == seed_num + 1] = 0
    cell_seg[cell_seg == seed_num + 2] = 0

    print("watershed based cell segmentation is done.")

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

    print("stack bottom has been properly updated.")

    ################################################################
    # part 8: QC by size
    ################################################################
    # remove small cells due to failure / noise
    for ii in np.unique(cell_seg[cell_seg > 0]):
        this_one_cell = cell_seg == ii
        this_dna = dna_mask_bw.copy()
        this_dna[this_one_cell == 0] = 0
        # small "cell" or "dna"
        if (
            np.count_nonzero(this_one_cell > 0) < 70000
            or np.count_nonzero(this_dna > 0) < 1000
            or np.count_nonzero(getLargestCC(this_dna, is_label=False)) < 10000
        ):
            cell_seg[this_one_cell > 0] = 0

    # false clip check (mem channel)
    # i.e. the segmentation should not appear in first of last z-slice
    z_range_mem = np.where(np.any(cell_seg, axis=(1, 2)))
    z_range_mem = z_range_mem[0]
    if (
        len(z_range_mem) == 0
        or z_range_mem[0] == 0
        or z_range_mem[-1] == cell_seg.shape[0] - 1
    ):
        print("exit because false clip or bad floaty is detected in mem channel")
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    print("size based QC is done")

    # relabel the index in case altered when dumping the bottom
    cell_seg, _tmp, _tmp2 = relabel_sequential(cell_seg.astype(np.uint8))

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
    if (
        len(z_range_dna) == 0
        or z_range_dna[0] == 0
        or z_range_dna[-1] == dna_mask_label.shape[0] - 1
    ):
        print("exit because false clip or bad floaty is detected in dna channel")
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    if dna_mask_label.max() < 3:  # if only a few cells left, just throw it away
        print("exit because only very few cells are segmented, maybe a bad image")
        if return_prediction:
            return None, [dna_mask_pred, mem_pred, seed_pred]
        else:
            return None

    print("refining dna masks ... ...")

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
            print("bug, empty dna is found, but should not")
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
            print("bug occurs in processing the pair ... ")
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
                    single_dna[zz, :, :] = remove_small_objects(
                        single_dna[zz, :, :], min_size=100
                    )
            single_dna = remove_small_objects(single_dna, min_size=2500)
            dna_mask_label[dna_mask_label == (cell_idx + 1)] = 0
            dna_mask_label[single_dna > 0] = cell_idx + 1
        else:
            # refine in each cell by removing small parts touching seperatinon bounary
            single_mem_bd = find_boundaries(cell_seg == (cell_idx + 1), mode="inner")
            bd_idx = list(np.unique(single_dna_label[single_mem_bd > 0]))
            if len(bd_idx) > 0:
                for list_idx, dna_bd_idx in enumerate(bd_idx):
                    if np.count_nonzero(single_dna_label == dna_bd_idx) < 600:
                        dna_mask_label[single_dna_label == dna_bd_idx] = 0

    print("refinement is done.")

    if return_prediction:
        return cell_seg, dna_mask_label, [dna_mask_pred, mem_pred, seed_pred]
    else:
        return [[cell_seg, dna_mask_label], ["cell_segmentation", "dna_segmentation"]]
