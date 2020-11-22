import numpy as np
from typing import List, Union
from pathlib import Path
from aicsmlsegment.utils import simple_norm
from aicsimageio import AICSImage


def SegModule(
    img: np.ndarray = None,
    model_list: List = None,
    filename: Union[str, Path] = None,
    index: Union[int, List[int]] = None,
    return_prediction: bool = False,
    pred_cutoff: float = 0.5
):
    """
    Segmentation function for CAAX segmentation.


    Parameters:
    ----------
    img: np.ndarray
        a 4D numpy array of size 1 x Z x Y x X of CAAX image
    filename: Union[str, Path]
        when img is None, use filename to load image
    index: Union[int, List[int]]
        an integers or a list of only one integer indicating which channel is CAAX.
        Only valid when using filename to load image. Not used when img is not None
    model_list: List
        the list of models to be applied on the image. Here, we assume 1 model
        is provided, CAAX segmentation model.
    return_prediction: book
        a flag indicating whether to return raw prediction
    pred_cutoff: float
        an empirically determined cutoff value to binarize the prediction from
        the CAAX segmentation model. Default is 0.5.

    Return:
    ------------
        one numpy array or together with raw prediction (if return_prediction is True)
    """

    if img is None:
        # load the image
        reader = AICSImage(filename)
        img = reader.data[0, index, :, :, :]

    # make sure the image has 4 dimensions
    assert len(img.shape) == 4

    # extract CAAX channel
    struct_img = img[0, :, :, :]
    struct_img = simple_norm(struct_img, 2, 11)

    # do prediction
    pred = model_list[0].apply_on_single_zstack(
        struct_img, already_normalized=True, cutoff=-1
    )

    bw = pred > pred_cutoff
    bw[:2, :, :] = 0
    bw[-2:, :, :] = 0

    bw = bw.astype(np.uint8)
    bw[bw > 0] = 255

    if return_prediction:
        return bw, pred
    else:
        return bw
