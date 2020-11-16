from tifffile import imsave
from shutil import copyfile
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage import gaussian_filter
import glob
import itk
import numpy as np
from skimage.morphology import watershed, remove_small_objects, remove_small_holes
from skimage.measure import label
from skimage.segmentation import find_boundaries, clear_border
from aicsmlsegment.utils import background_sub, simple_norm
from aicsimageio import AICSImage, omeTifWriter

mitosis_cutoff = 0.5  # 0.3
core_cutoff = 0.5
fill_est_cutoff = 0.5
celledge_cutoff = 0.95
min_overlap = 0.9
min_core = 1500

max_size_dna = 1000000
min_size_dna = 2000

precut_th = 0.25


def SegModule(
    img=None, model_list=None, filename=None, index=None, return_prediction=False
):

    # model order: mitosis, fill, core, mem_edge
    if img is None:
        # load the image
        reader = AICSImage(filename)
        img = reader.data[0, index, :, :, :]

    # make sure the image has 4 dimensions
    assert len(img.shape) == 4 and img.shape[0] == 1

    # extract lamin channel
    lamin_img = img[0, :, :, :]
    # lamin_img[lamin_img>4000] = lamin_img.min()
    lamin_img = background_sub(lamin_img, 50)

    # model 1: fill

    fill_pred = model_list[0].apply_on_single_zstack(
        lamin_img, already_normalized=True, cutoff=-1
    )
    fill_smooth = gaussian_filter(fill_pred, sigma=1, mode="nearest", truncate=3)
    pre_mask = fill_smooth > precut_th  # 0.7
    pre_mask = remove_small_objects(pre_mask, min_size=min_size_dna)

    """
    writer = omeTifWriter.OmeTifWriter('fillpred.tiff')
    writer.save(fill_pred.astype(np.float32))
    writer = omeTifWriter.OmeTifWriter('smooth.tiff')
    writer.save(fill_smooth.astype(np.float32))
    writer = omeTifWriter.OmeTifWriter('premask.tiff')
    writer.save(pre_mask.astype(np.uint8))
    """

    # model 2: core
    core_pred = model_list[1].apply_on_single_zstack(
        lamin_img, already_normalized=True, cutoff=-1
    )
    core_bw = core_pred > core_cutoff
    core_bw[:2, :, :] = 0
    core_bw[-2:, :, :] = 0
    remove_small_objects(core_bw, min_size=min_core, in_place=True)

    # writer = omeTifWriter.OmeTifWriter('core.tiff')
    # writer.save(core_pred.astype(np.float32))

    # watershed
    seed = label(core_bw)
    final = watershed(
        1.01 - fill_smooth, seed.astype(int), mask=pre_mask, watershed_line=True
    )

    # finalize
    boundary_mask = np.zeros_like(core_bw)
    boundary_mask[:, :5, :] = 1
    boundary_mask[:, -5:, :] = 1
    boundary_mask[:, :, :5] = 1
    boundary_mask[:, :, -5:] = 1

    bd_idx = list(np.unique(final[boundary_mask > 0]))

    num_cell = final.max()
    final_bw = np.zeros_like(final)
    total_cell = 0
    for cid in range(num_cell):
        if (cid + 1) in bd_idx:
            continue

        tmp_cell = final == (cid + 1)
        for zz in range(tmp_cell.shape[0]):
            tmp_z = tmp_cell[zz, :, :]
            if np.any(tmp_z):
                tmp_cell[zz, :, :] = remove_small_objects(
                    binary_fill_holes(tmp_z), min_size=100
                )

        cell_size = np.count_nonzero(tmp_cell > 0)
        if cell_size > max_size_dna or cell_size < min_size_dna:
            continue

        total_cell += 1
        final_bw[tmp_cell > 0] = total_cell

    num_cell = final_bw.max()
    if num_cell > 200:
        final_bw = final_bw.astype(np.uint8)
    else:
        final_bw = final_bw.astype(np.uint16)

    if return_prediction:
        return [final_bw], ["_fill_"], [fill_pred, core_pred], ["_fill_", "_core_"]
    else:
        return [final_bw], ["_fill_"]
