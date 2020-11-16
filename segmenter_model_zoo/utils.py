import numpy as np

# import sys
from pathlib import Path
from typing import Union
import os
from glob import glob
from skimage.measure import label
from aicsimageio.writers import OmeTiffWriter
import re


def save_as_uint(
    img: np.ndarray,
    save_path: Union[str, Path],
    core_fn: str,
    tag: str = "segmentation",
    overwrite: bool = False,
):
    """
    save the segmentation to disk

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


def load_filenames(data_config):

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
            all_timelapse.extend(str(data_item["dir"]))
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
        quit()
    elif len(all_stacks) == 0 and len(all_timelapse) == 0:
        print("not file is found")
        quit()

    if len(all_stacks) > 0:
        return all_stacks, False
    else:
        print("timelapse is not supported yet")
        # return all_timelapse, True
