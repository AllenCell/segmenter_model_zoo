import numpy as np

import sys
import os
from pathlib import Path
from typing import Union
from glob import glob
from skimage.measure import label
from aicsimageio.writers import OmeTiffWriter
import re
from collections import Counter


################################################################################
# common util functions 
################################################################################
def getLargestCC(labels, is_label=True):
    """
    return the largest connect component from a label image or a binary image
    """
    if is_label:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        sub_labels = label(labels > 0, connectivity=3, return_num=False)
        largestCC = sub_labels == np.argmax(np.bincount(sub_labels.flat)[1:]) + 1

    return largestCC


################################################################################
# util functions for io
################################################################################
def save_as_uint(
    img: np.ndarray,
    save_path: Union[str, Path],
    core_fn: str,
    tag: str = "segmentation",
    overwrite: bool = False,
):
    """
    save the segmentation to disk as uint type 
    (either 8-bit or 16-bit depending on data)

    Parameters
    ---
    img: np.ndarray
        the image to save

    save_path: Union[str, Path]
        the path to save the image

    core_fn: str
        saved filename will be {core_fn}_{tag}.tiff

    tag: str
        the tag to be added to the end of output filename
        default is "segmentation"
    """

    # find the minimal bit
    if img.max() < 255:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.uint16)

    # save the file
    save_path = Path(save_path).expanduser().resolve(strict=True)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    with OmeTiffWriter(
        save_path / f"{core_fn}_{tag}.tiff", overwrite_file=overwrite
    ) as writer:
        writer.save(img)


def load_filenames(data_config):
    """
    load the filenames of all the images need to be processed based on config.
    Three types of loading are currently supported:
    - "folder"
        * two parameters are required: "dir" and "search". Basically, we just
        search using the rules defined by "search" (either "*" for all files
        or a regular expression) inside the folder "dir". 
        * one optional parameter: "data_range". If provided, a specific range
        of all files will be processed, e.g. [0, 100] or [100, -1]. This is 
        useful when running in a distributed setting. 
    - "zstack"
        * one parameter: "dir" specifying the filepath
    - "csv"
        * two parameters are required: "file" and "column". After loading the 
        csv pointed by "file", all filenames in the column defined by "column"
        will be identified as files to be processed.
    """

    all_stacks = []
    all_timelapse = []
    for data_item in data_config:
        if data_item["item"] == "folder":

            if data_item["search"] == "*":
                filenames = glob(data_item["dir"] + os.sep + "/*")
            else:
                reg = re.compile(data_item["search"])
                filenames = [
                    data_item["dir"] + os.sep + f
                    for f in os.listdir(data_item["dir"])
                    if reg.search(f)
                ]
            filenames.sort()
            # filenames = sorted(filenames, reverse=True)
            if "data_range" in data_item:
                if data_item["data_range"][1] == -1:
                    all_stacks.extend(filenames[data_item["data_range"][0] :])
                else:
                    all_stacks.extend(
                        filenames[
                            data_item["data_range"][0] : data_item["data_range"][1]
                        ]
                    )
            else:
                all_stacks.extend(filenames)
        elif data_item["item"] == "zstack":
            all_stacks.extend([data_item["dir"]])
        elif data_item["item"] == "timelapse":
            print("timelapse is not supported yet")
            sys.exit(0)
            # TODO
            # all_timelapse.extend(str(data_item["dir"]))
        elif data_item["item"] == "csv":
            import pandas as pd

            df = pd.read_csv(data_item["file"])
            col = str(data_item["column"])
            for _index, row in df.iterrows():
                fn = row[col]
                assert os.path.exists(fn)
                all_stacks.extend([fn])

    if len(all_stacks) > 0 and len(all_timelapse) > 0:
        print("we cannot handle timelapse data and non-timelapse data in one shot")
        print("please do this separately")
        sys.exit(0)
    elif len(all_stacks) == 0 and len(all_timelapse) == 0:
        print("not file is found")
        sys.exit(0)

    if len(all_stacks) > 0:
        return all_stacks, False
    else:
        print("not file is found")
        sys.exit(0)


################################################################################
# util functions for dna and cell segmentations
################################################################################
def bb_intersection_over_union(boxA, boxB):
    """computer IOU of two bounding boxes"""

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
    """find the most likely pair from all candidate in one bounding box"""

    # collect confidence score of all candidates 
    score = []
    for ii in range(len(main_score)):
        score.append((main_score[ii], aux_score[ii]))

    # sort
    score_array = np.array(score, dtype="<f4,<f4")
    weight_order = score_array.argsort()
    return weight_order.argmax()


def exist_double_assignment(cell_pairs):
    """check if one object is associated with multiple pairs"""

    nodes = set(x for ll in cell_pairs for x in ll)
    num_node = len(nodes)
    num_pair = len(cell_pairs)

    return not num_pair * 2 == num_node


def find_multi_assignment(cell_pairs):
    """find all multi-assignment"""

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
    """update candidates after taking out one pair"""

    p_best = multi_pair[current_best_pair]
    idx_to_remove = []
    for idx, p in enumerate(multi_pair):
        if idx == current_best_pair:
            continue
        if p[0] in p_best or p[1] in p_best:
            idx_to_remove.append(idx)

    return idx_to_remove
