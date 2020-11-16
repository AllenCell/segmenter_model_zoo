import numpy as np
from aicsmlsegment.utils import simple_norm
from aicsimageio import AICSImage

pred_cutoff = 0.5


def SegModule(
    img=None, model_list=None, filename=None, index=None, return_prediction=False
):

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
