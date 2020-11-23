import sys
from typing import List, Union
from pathlib import Path
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import gaussian_filter
import itk
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.segmentation import find_boundaries
from aicsmlsegment.utils import background_sub, simple_norm
from aicsimageio import AICSImage

mitosis_cutoff = 0.5  # 0.3
core_cutoff = 0.5
fill_est_cutoff = 0.5
celledge_cutoff = 0.95
min_overlap = 0.9
min_core = 1600


def SegModule(
    img: np.ndarray = None,
    model_list: List = None,
    filename: Union[str, Path] = None,
    index: List[int] = None,
    return_prediction: bool = False,
    output_type: str = "production",
):
    """
    Segmentation function for lamin b1 morphological segmentation. The name
    "morphological segmentation" refers that the lamin shells are guaranteed
    to be a fully closed shell (i.e. topologically fillable in 3D)


    Parameters:
    ----------
    img: np.ndarray
        a 4D numpy array of size 2 x Z x Y x X, the first channel is lamin b1
        and the second channel is cell membrane.
    filename: Union[str, Path]
        when img is None, use filename to load image
    index: List[int]
        a list of 2 integers, the first indicating which channel is lamin b1,
        the second integer indicating which channel is cell membrane. Only
        valid when using filename to load image. Not used when img is not None
    model_list: List
        the list of models to be applied on the image. Here, we assume 4 models
        are provided (in this specific order): lamin structure segmentation model,
        lamin fill model, lamin core model, and membrane segmentation model.
    return_prediction: book
        a flag indicating whether to return raw prediction
    output_type: str
        There are two ways to return the output: "production" (default) and "RnD".
        "production" means only the final lamin b1 segmentation result will be
        returned. "RnD" means lamin fill and lamin shell will also be returned,
        which are intermediate results for generating the final lamin b1 segmentation.
        lamin fill is usefull particularly to represent nuclear segmentation and
        can be used to measured the nuclei shapes. lamin shell is just the boundary
        of lamin fill.

    Return:
    ------------
        1 or 3 numpy array (depending on output_type) or together with raw
        prediction (if return_prediction is True)
    """

    # model order: mitosis, fill, core, mem_edge
    if img is None:
        # load the image
        reader = AICSImage(filename)
        img = reader.data[0, index, :, :, :]

    # make sure the image has 4 dimensions
    assert len(img.shape) == 4 and img.shape[0] == 2

    # extract lamin channel
    lamin_img = img[0, :, :, :]
    lamin_img[lamin_img > 4000] = lamin_img.min()
    lamin_img = background_sub(lamin_img, 50)

    # extra cellmask channel
    mem_img = img[1, :, :, :]
    mem_img[mem_img > 10000] = mem_img.min()
    mem_img = background_sub(mem_img, 50)
    mem_img = simple_norm(mem_img, 2, 11)

    # model 1: mitosis
    mitosis_pred = model_list[0].apply_on_single_zstack(
        lamin_img, already_normalized=True, cutoff=-1
    )
    mitosis_smooth = gaussian_filter(mitosis_pred, sigma=1, mode="nearest", truncate=3)

    # import random
    # rr = random.randint(100,199)
    # imsave('test_all_'+str(rr)+'.tiff', mitosis_pred)

    # model 2: fill
    fill_pred = model_list[1].apply_on_single_zstack(
        lamin_img, already_normalized=True, cutoff=-1
    )
    fill_estimation = fill_pred > fill_est_cutoff

    # imsave('test_fill_'+str(rr)+'.tiff', fill_pred)

    # model 3: core
    core_pred = model_list[2].apply_on_single_zstack(
        lamin_img, already_normalized=True, cutoff=-1
    )
    core_bw = core_pred > core_cutoff
    core_bw = remove_small_objects(core_bw, min_size=min_core)

    # imsave('test_core_'+str(rr)+'.tiff', core_bw.astype(np.uint8))

    # model 4: cell mask to further prune the mask
    cellmask_pred = model_list[3].apply_on_single_zstack(
        mem_img, already_normalized=True, cutoff=-1
    )
    cellmask_bw = cellmask_pred > celledge_cutoff

    # imsave('test_edge_'+str(rr)+'.tiff', cellmask_pred)

    # prepare for watershed
    mitosis_smooth[cellmask_bw > 0] = 0.005

    # do watershed
    seed, seed_num = label(core_bw, return_num=True)
    seed[cellmask_bw > 0] = seed_num + 1

    raw_itk = itk.GetImageFromArray(mitosis_smooth)
    seed_itk = itk.GetImageFromArray(seed.astype(np.int16))
    seg_itk = itk.morphological_watershed_from_markers_image_filter(
        raw_itk, marker_image=seed_itk, fully_connected=True, mark_watershed_line=False
    )
    filled = itk.GetArrayFromImage(seg_itk)

    # filled = watershed(mitosis_smooth, seed, watershed_line=False)

    filled[filled > seed_num] = 0

    fill_ref = np.logical_or(
        fill_estimation, binary_fill_holes(mitosis_pred > mitosis_cutoff)
    )
    # imsave('test_new_fill_'+str(rr)+'.tiff', filled.astype(np.uint8))
    # imsave('test_fill_ref_'+str(rr)+'.tiff', fill_ref.astype(np.uint8))

    # remove "failed" cells (certain mitotic stage is not fillable, or is filled badly)
    for ci in range(seed_num):
        single_fill = filled == ci + 1
        if (
            np.count_nonzero(np.logical_and(single_fill, fill_estimation == 0))
            > 5000000
        ):
            filled[single_fill > 0] = 0
            continue

        overlap_area = np.logical_and(single_fill, fill_ref)
        overlap_ratio = np.count_nonzero(overlap_area) / (
            1e-8 + np.count_nonzero(single_fill)
        )
        if overlap_ratio < min_overlap:
            filled[single_fill > 0] = 0

    # get two versions of segmentation
    shell = find_boundaries(filled, mode="outer")
    merged = np.logical_or(shell > 0, mitosis_pred > mitosis_cutoff)

    # return results
    merged = merged.astype(np.uint8)
    merged[merged > 0] = 255
    merged[0, :, :] = 0
    merged[-1, :, :] = 0

    shell = shell.astype(np.uint8)
    shell[shell > 0] = 255

    if output_type == "production":
        return merged
    elif output_type == "RnD":
        if return_prediction:
            return (
                [merged, filled.astype(np.uint8), shell],
                ["_structure_", "_fill_", "_shell_"],
                [mitosis_pred, fill_pred, core_pred, cellmask_pred],
                ["_all_", "_fill_", "_core_", "_edge_"],
            )
        else:
            return [merged, filled.astype(np.uint8), shell], [
                "_structure_",
                "_fill_",
                "_shell_",
            ]
    else:
        print("unsupported output type")
        sys.exit(0)
