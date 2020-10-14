import numpy as np
from skimage.morphology import remove_small_objects, disk, erosion, watershed, ball, dilation, binary_opening
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from aicsmlsegment.utils import background_sub, simple_norm
from aicsimageprocessing import resize
from skimage.io import imsave
from aicsimageio import omeTifWriter

def SegModule(input_img, models, model_name, return_prediction=False):
    
    if model_name == 'Nucleus_instance_hipsc_880':
        min_size_dna = 100000 
        max_size_dna = 2000000
    elif model_name == 'Nucleus_instance_hipsc_ZSD':
        min_size_dna = 40000 #90000
        max_size_dna = 700000
    elif model_name == 'Nucleus_instance_cardio_ZSD': 
        min_size_dna = 100000 
        max_size_dna = 1600000
    elif model_name == 'Nucleus_instance_cardio_880':
        min_size_dna = 100000 
        max_size_dna = 9000000

    # assume 4 d image: 1 x Z x Y x X
    assert len(input_img.shape)==4 and input_img.shape[0]==1

    # assume two models: mask, seed
    assert len(models)==2

    # normalization
    img = background_sub(input_img[0,:,:,:],50)
    img = simple_norm(img, 1.5, 10)
    #img = simple_norm(input_img[0,:,:,:], 2.5, 10)

    # rescale by default
    #img = resize(img, (1, 0.5, 0.5), method='cubic')
    #img = ( img - img.min() ) / (img.max() - img.min() + 1e-8)

    # get mask segmentation
    bw1 = models[0].apply_on_single_zstack(img, already_normalized=True, cutoff=-1) 
    bw1 = bw1 > models[0].get_cutoff()

    # get core segmentation 
    bw2 = models[1].apply_on_single_zstack(img, already_normalized=True, cutoff=-1) 
    bw2 = remove_small_objects(bw2 > models[1].get_cutoff(), min_size = 100)
    bw1 = np.logical_or(bw1>0, bw2>0)

    # remove all the objects that have no seed detected in bw2
    bw1 = remove_small_objects(bw1, min_size=1600)
    lab1 = label(bw1>0)
    cid = np.unique(lab1[bw2>0])

    seg = np.zeros_like(bw1)
    for ii in range(len(cid)):
        idx = cid[ii]
        if idx>0:
            seg[lab1==idx]=1

    # do seeded watershed
    seed, seed_num = label(bw2, return_num=True)
    dist = np.zeros(bw1.shape)
    for zz in range(bw1.shape[0]):
        tmp = bw1[zz,:,:]>0
        if np.any(tmp):
            dist[zz,:,:] = distance_transform_edt(tmp)

    final = watershed(-dist, seed.astype(int), mask=bw1, watershed_line=True)
    
    boundary_mask = np.zeros_like(seg)
    boundary_mask[:, :5, :] = 1
    boundary_mask[:,-5:, :] = 1
    boundary_mask[:, :, :5] = 1
    boundary_mask[:, :,-5:] = 1

    bd_idx = list(np.unique(final[boundary_mask>0]))

    num_cell = final.max()
    final_bw = np.zeros_like(final)
    total_cell = 0
    for cid in range(num_cell):
        if (cid+1) in bd_idx:
            continue

        tmp_cell = final==(cid+1)
        for zz in range(tmp_cell.shape[0]):
            tmp_z = tmp_cell[zz,:,:]
            if np.any(tmp_z):
                tmp_z = binary_opening(binary_fill_holes(tmp_z), selem=disk(5))
                tmp_cell[zz,:,:] = remove_small_objects(tmp_z, min_size = 100)
            
        #tmp_cell = dilation(tmp_cell, selem=ball(1))

        cell_size = np.count_nonzero(tmp_cell>0)
        if cell_size > max_size_dna or cell_size < min_size_dna:
            print(cell_size)
            continue

        # check if this cell should be kept
        total_cell += 1
        '''
        if model_name == 'Nucleus_instance_hipsc_ZSD':
            tmp_cell = dilation(tmp_cell>0,selem=ball(1))
            for zz in range(tmp_cell.shape[0]):
                tmp_z = tmp_cell[zz,:,:]
                if np.any(tmp_z):
                    tmp_cell[zz,:,:] = dilation(tmp_z, selem=disk(3))
                    
            final_bw[tmp_cell] = total_cell
        else:
            final_bw[tmp_cell>0] = total_cell
        '''
        final_bw[tmp_cell>0] = total_cell

    if return_prediction:
        return final_bw, [bw1, bw2]
    else:
        return final_bw
