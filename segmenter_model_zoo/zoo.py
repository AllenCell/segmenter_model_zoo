import os
import logging
import numpy as np

import torch
from torch.autograd import Variable
from aicsmlsegment.utils import input_normalization
from aicsimageprocessing import resize
from aicsimageio import AICSImage
import time

from fnet.models import load_model as load_model_lf
from fnet.cli.predict import parse_model

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

MODEL_DEF_MAPPING = {
    'unet_xy_zoom': {'size_in':[52, 420, 420], 'size_out':[20, 152, 152], 'nclass': [2,2,2], 'nchannel':1, 'OutputCh':[0,1]},
    'unet_xy': {'size_in':[44, 140, 140], 'size_out':[16, 52, 52], 'nclass': [2,2,2], 'nchannel':1, 'OutputCh':[0,1]}
}

CHECKPOINT_PATH_MAPPING = {
    'H2B_production_alpha':{'model_type':'unet_xy_zoom', 'norm':18, 'path':'/allen/aics/assay-dev/Jianxu/ThemedProjects/H2B_dl_segmentation/models/20190623/checkpoint_epoch_400.pytorch','default_cutoff':0.3},
    'DNA_mask_production_alpha':{'model_type':'unet_xy_zoom', 'norm':12, 'path':'/allen/aics/assay-dev/users/Jianxu/data/mitotic_data_samples/models/dye_mitotic_20190915/checkpoint_epoch_520.pytorch','default_cutoff':0.4},
    'DNA_seed_production_alpha':{'model_type':'unet_xy_zoom', 'norm':12, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/DNA/dye_seed_20190620/checkpoint_epoch_1000.pytorch','default_cutoff':0.9},
    'CellMask_edge_production_alpha':{'model_type':'unet_xy_zoom', 'norm':13, 'path':'/allen/aics/assay-dev/users/Jianxu/data/training_assay_caax_cellmask/models/cellmask_transfer_new_segmenter/checkpoint_epoch_400.pytorch','default_cutoff':0.5},
    'CAAX_production_alpha':{'model_type':'unet_xy_zoom', 'norm':1, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/caax/20181003_02/unet_xy_p3-600-default.pth','default_cutoff':0.25},
    'CAAX_production_beta':{'model_type':'unet_xy_zoom', 'norm':13, 'path':'/allen/aics/assay-dev/users/Jianxu/data/training_assay_caax_cellmask/models/intial_caax_better_top/checkpoint_epoch_950.pytorch','default_cutoff':0.25},
    'LMNB1_all_production': {'model_type':'unet_xy', 'norm':15, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/LAMINB1/20181117_01/unet_xy-200-default.pth','default_cutoff':0.5},
    'LMNB1_fill_production_alpha': {'model_type':'unet_xy_zoom', 'norm':15, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/LAMINB1/20190513_01/checkpoint_epoch_300.pytorch','default_cutoff':0.5},
    'LMNB1_core_production_alpha': {'model_type':'unet_xy_zoom', 'norm':15, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/LAMINB1/20190513_02/checkpoint_epoch_300.pytorch','default_cutoff':0.5},
    'Nucleus_cardio_880_mask':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181115_01/unet_xy_p3-160-default.pth','default_cutoff':0.5},
    'Nucleus_cardio_880_seed':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181115_02/unet_xy_p3-160-default.pth','default_cutoff':0.5},
    'Nucleus_cardio_ZSD_mask':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181031_01/unet_xy_p3-200-default.pth','default_cutoff':0.5},
    'Nucleus_cardio_ZSD_seed':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181031_02/unet_xy_p3-200-default.pth','default_cutoff':0.5},
    'Nucleus_hipsc_880_mask':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181119_01/unet_xy_p3-200-default.pth','default_cutoff':0.5},
    'Nucleus_hipsc_880_seed':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/H2B/20181119_02/unet_xy_p3-200-default.pth','default_cutoff':0.5},
    'Nucleus_hipsc_ZSD_mask':{'model_type':'unet_xy_zoom', 'norm':18, 'path':'/allen/aics/assay-dev/Jianxu/ThemedProjects/H2B_dl_segmentation/models/20190623/checkpoint_epoch_400.pytorch','default_cutoff':0.3},
    'Nucleus_hipsc_ZSD_seed':{'model_type':'unet_xy_zoom', 'norm':2, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/DNA/20181001_02/unet_xy_p3-400-default.pth','default_cutoff':0.5},
    'CAAX_production':{'model_type':'unet_xy_zoom', 'norm':1, 'path':'//allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/caax/20181003_02/unet_xy_p3-600-default.pth','default_cutoff':0.5},
    'CAAX_mask':{'model_type':'unet_xy_zoom', 'norm':1, 'path':'//allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/caax/20181003_01/unet_xy_p3-600-default.pth','default_cutoff':0.5},
    'LF_DNA_mask':{'path':'/allen/aics/assay-dev/users/Jianxu/Projects/labelfree_rnd/models/dna_mask/model.p','default_cutoff':1.4},
    'LF_DNA_mask_two_camera':{'path':'/allen/aics/assay-dev/users/Jianxu/data/two_camera_label_free/models/bf_dna/model.p','default_cutoff':1.1},
    'LF_DNA_fluorescent':{'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/labelfree_analysis/models/dna/model.p','default_cutoff':1.4},
    'LF_mem_edge':{'path':'/allen/aics/assay-dev/users/Jianxu/Projects/labelfree_rnd/models/mem_edge/model.p','default_cutoff':1.4},
    'LF_mem_edge_two_camera':{'path':'/allen/aics/assay-dev/users/Jianxu/data/two_camera_label_free/models/bf_mem/model.p','default_cutoff':1.4},
    'DNA_mask_from_BF':{'model_type':'unet_xy_zoom', 'norm':11, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/DNA/20190415_01/unet_xy_p3-100-default.pth','default_cutoff':0.5},
    'DNA_core_from_BF':{'model_type':'unet_xy_zoom', 'norm':11, 'path':'/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/DNA/20190415_02/unet_xy_p3-100-default.pth','default_cutoff':0.85},
}


##### HACK #####
# backup models:
# DNA_Mask: '/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/DNA/dye_all_20190605/checkpoint_epoch_500.pytorch', with cutoff = 0.7
# cellmask: ''/allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/CellMask/20190503_01/checkpoint_epoch_750.pytorch'. with cutoff = 0.5
# H2b mask: '/allen/aics/assay-dev/Jianxu/ThemedProjects/H2B_dl_segmentation/models/20190623/checkpoint_epoch_400.pytorch'
s_H2B_instance_hipsc_ZSD_40x = """*********************
use as function:
    input can be 2 x Z x Y x X 
use as batch processing:
    input  
**************************"""

s_Nucleus_instance_hipsc_880 = """*********************
Note: This model is not fully optmized. The accuracy is comparable to 
    the results obtained in Nov-Dec 2018 for ASCB poster. 

expected output:
    For each input, say XYZ123.czi, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will be automatically removed. The 
    model cannot guarantee every cell is correctly segmented. A manual curation step
    is encouraged and indeed used for 2018 ASCB poster.
expected input image meta:
    structure: H2B, CBX1 in hIPS cells
    modality: AiryScan fast (880)
    condition: Control, DMSO, TSA
    resolution (micron): 0.2222 (Z) x 0.0497 (Y) x 0.0497 (X)

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [0])
**************************"""

s_Nucleus_instance_hipsc_ZSD = """*********************
Note: This model is not fully optmized. The accuracy is comparable to 
    the results obtained in Nov-Dec 2018 for ASCB poster. 

expected output:
    For each input, say XYZ123.czi, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will be automatically removed. The 
    model cannot guarantee every cell is correctly segmented. A manual curation step
    is encouraged and indeed used for 2018 ASCB poster.
expected input image meta:
    structure: H2B, CBX1 in hIPS cells
    modality: ZSD
    condition: Control, DMSO, TSA
    resolution (micron): 0.29 (Z) x 0.108 (Y) x 0.108 (X)

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [0])
**************************"""

s_Nucleus_instance_cardio_ZSD = """*********************
Note: This model is not fully optmized. The accuracy is comparable to 
    the results obtained in Nov-Dec 2018 for ASCB poster. 

expected output:
    For each input, say XYZ123.czi, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will be automatically removed. The 
    model cannot guarantee every cell is correctly segmented. A manual curation step
    is encouraged and indeed used for 2018 ASCB poster.
expected input image meta:
    structure: H2B, CBX1 in cardio cells
    modality: ZSD
    condition: Control, DMSO, TSA
    resolution (micron): 0.29 (Z) x 0.108 (Y) x 0.108 (X)

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [0])
**************************"""


s_Nucleus_instance_cardio_880 = """*********************
Note: This model is not fully optmized. The accuracy is comparable to 
    the results obtained in Nov-Dec 2018 for ASCB poster. 

expected output:
    For each input, say XYZ123.czi, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will be automatically removed. The 
    model cannot guarantee every cell is correctly segmented. A manual curation step
    is encouraged and indeed used for 2018 ASCB poster.
expected input image meta:
    structure: H2B, CBX1 in cardio cells
    modality: AiryScan fast (880)
    condition: Control, DMSO, TSA
    resolution (micron): 0.2222 (Z) x 0.0497 (Y) x 0.0497 (X)

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [0])
**************************"""

s_LF_DNA_instance = """*********************
Note: This model is not fully optmized. The accuracy is NOT guaranteed

expected output:
    For each input, say XYZ123.czi, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will be automatically removed. The 
    model cannot guarantee every cell is correctly segmented. A manual curation step
    is encouraged.
expected brightfield quality: comparable to pipeline 4

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [-1])
**************************"""

s_DNA_instance_from_lf = """*********************
Note: This model is not fully optmized. The accuracy is NOT guaranteed

expected output:
    For each input, say XYZ123.tiff, the output will be a tiff image (uint8)
    named as XYZ123_segmentation.tiff of the same x/y/z size as the input 
    image. The results are not binary. Each cell is labeled by a unique integer. 
    The cells touching the image boundary will not be removed. The model cannot 
    guarantee every cell is correctly segmented. A manual curation step is 
    encouraged.

use as function:
    option 1: input as numpy array of size 1 x Z x Y x X
    option 2: input as filename and channel index list, [ch_index]
use as batch processing: 
    double check following paramters in the yaml file
        channel index: [ch_index] (e.g., [0])
**************************"""


SUPER_MODEL_MAPPING = {
    'DNA_MEM_instance_production_alpha':{'models':['DNA_mask_production_alpha','CellMask_edge_production_alpha','DNA_seed_production_alpha'],\
        'instruction':'Reminder:input can be 2 x Z x Y x X, where first channel is dna, second is mem. Or file path and index list [dna_index, mem_index]'},
    'DNA_MEM_instance_LF_integration':{'models':['DNA_mask_production_alpha','CellMask_edge_production_alpha','DNA_seed_production_alpha','LF_DNA_mask','LF_mem_edge'],\
        'instruction':'Reminder:input can be 3 x Z x Y x X, where first channel is dna, second is mem, third is bf. Or file path and index list [dna_index, mem_index, bf_index]'},
    'DNA_MEM_instance_LF_integration_two_camera':{'models':['DNA_mask_production_alpha','CellMask_edge_production_alpha','DNA_seed_production_alpha','LF_DNA_mask_two_camera','LF_mem_edge_two_camera'],\
        'instruction':'Reminder:input can be 3 x Z x Y x X, where first channel is dna, second is mem, third is bf. Or file path and index list [dna_index, mem_index, bf_index]'},
    'DNA_instance_LF_integration_two_camera':{'models':['DNA_mask_production_alpha','DNA_seed_production_alpha','LF_mem_edge_two_camera'],\
        'instruction':'Reminder:input can be 3 x Z x Y x X, where first channel is dna, third is bf. Or file path and index list [dna_index, bf_index]'},
    'LMNB1_morphological_production_alpha':{'models':['LMNB1_all_production','LMNB1_fill_production_alpha','LMNB1_core_production_alpha','CellMask_edge_production_alpha'],\
        'instruction':'Reminder:input must be 2 x Z x Y x X (4d array) or file path and index of the lamin b1 channel and cellmask channel'},
    'LMNB1_morphological_RnD':{'models':['LMNB1_all_production','LMNB1_fill_production_alpha','LMNB1_core_production_alpha','CellMask_edge_production_alpha'],\
        'instruction':'Reminder:input must be 2 x Z x Y x X (4d array) or file path and index of the lamin b1 channel and cellmask channel'},
    'LMNB1_fill_instance_100x_hipsc':{'models':['LMNB1_fill_production_alpha','LMNB1_core_production_alpha'],\
        'instruction':'Reminder:input must be 1 x Z x Y x X (4d array) or file path and index of the lamin b1 channel'},
    'LMNB1_morphological_with_labelfree':{'models':['LMNB1_all_production','LMNB1_fill_production_alpha','LMNB1_core_production_alpha','LF_mem_edge_two_camera'],\
        'instruction':'Reminder:input must be 2 x Z x Y x X (4d array) or file path and index of the lamin b1 channel and bright field channel'},
    'H2B_instance_hipsc_ZSD_40x': {'models':['H2B_40x', 'H2B_40x_seed'],\
        'instruction':s_H2B_instance_hipsc_ZSD_40x},
    'Nucleus_instance_hipsc_880': {'models':['Nucleus_hipsc_880_mask','Nucleus_hipsc_880_seed'],\
        'instruction':s_Nucleus_instance_hipsc_880},
    'Nucleus_instance_hipsc_ZSD': {'models':['Nucleus_hipsc_ZSD_mask','Nucleus_hipsc_ZSD_seed'],\
        'instruction':s_Nucleus_instance_hipsc_ZSD},
    'Nucleus_instance_cardio_ZSD': {'models':['Nucleus_cardio_ZSD_mask','Nucleus_cardio_ZSD_seed'],\
        'instruction':s_Nucleus_instance_cardio_ZSD},
    'Nucleus_instance_cardio_880': {'models':['Nucleus_cardio_880_mask','Nucleus_cardio_880_seed'],\
        'instruction':s_Nucleus_instance_cardio_880},
    'LF_DNA_instance_alpha': {'models':['LF_DNA_mask','LF_mem_edge'],\
        'instruction':s_LF_DNA_instance},
    'DNA_instance_from_labelfree_alpha': {'models':['DNA_mask_from_BF','DNA_core_from_BF'],\
        'instruction':s_DNA_instance_from_lf},
    'structure_H2B_production': {'models':['H2B_production_alpha'],\
        'instruction':'to be added'},
    'structure_AAVS1_production': {'models':['CAAX_production_alpha'],\
        'instruction':'to be added'},
    'DNA_MEM_instance_CAAX_with_BF':{'models':['CellMask_edge_production_alpha','LF_DNA_mask'],\
        'instruction':'Reminder:input can be 2 x Z x Y x X, where first channel is CAAX, second is bf. Or file path and index list [caax_index, bf_index]'},
}

class SegModel:

    def __init__(self):

        self.model = None
        self.normalization = None
        self.size_in = None 
        self.size_out = None 
        self.nclass = None 
        self.nchannel = None 
        self.OutputCh = None 
        self.cutoff = None

    def reset(self):

        self.model = None
        self.normalization = None
        self.size_in = None 
        self.size_out = None 
        self.nclass = None 
        self.nchannel = None 
        self.OutputCh = None
        self.cutoff = None 

    def to_gpu(self, gpu_id):
        if self.model is None:
            print('load the model first')
            quit()
        self.model = self.model.to(gpu_id)

    def load_train(self, checkpoint_name, model_param={}):

        if not (checkpoint_name in CHECKPOINT_PATH_MAPPING):
            raise IOError(f"Checkpoint '{checkpoint_name}' does not exist")

        if checkpoint_name[:2] == 'LF': # labelfree model
            model_def = parse_model(CHECKPOINT_PATH_MAPPING[checkpoint_name]['path'])
            model = load_model(model_def['path'], no_optim=True)
            self.model = model

        else:
            model_type = CHECKPOINT_PATH_MAPPING[checkpoint_name]['model_type']

            # load default model parameters or from model_param
            if 'size_in' in model_param:
                self.size_in = model_param['size_in']
            else: 
                self.size_in = MODEL_DEF_MAPPING[model_type]['size_in']
            
            if 'size_out' in model_param:
                self.size_out = model_param['size_out']
            else: 
                self.size_out = MODEL_DEF_MAPPING[model_type]['size_out']

            if 'nclass' in model_param:
                self.nclass = model_param['nclass']
            else: 
                self.nclass = MODEL_DEF_MAPPING[model_type]['nclass']

            if 'nchannel' in model_param:
                self.nchannel = model_param['nchannel']
            else: 
                self.nchannel = MODEL_DEF_MAPPING[model_type]['nchannel']

            if 'OutputCh' in model_param:
                self.OutputCh = model_param['OutputCh']
            else: 
                self.OutputCh = MODEL_DEF_MAPPING[model_type]['OutputCh']

            # define the model
            if  model_type == 'unet_xy':
                from aicsmlsegment.Net3D.unet_xy import UNet3D as DNN
                model = DNN(self.nchannel, self.nclass)
                self.softmax = model.final_activation
            elif model_type =='unet_xy_zoom':
                from aicsmlsegment.Net3D.unet_xy_enlarge import UNet3D as DNN
                if 'zoom_ratio' in model_param:
                    zoom_ratio = model_param['zoom_ratio']
                else:
                    zoom_ratio = 3
                model = DNN(self.nchannel, self.nclass, zoom_ratio)
                self.softmax = model.final_activation
            else: 
                print(f'name error')
                quit()
            self.model = model

            # load the trained model
            #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
            state = torch.load(CHECKPOINT_PATH_MAPPING[checkpoint_name]['path'], map_location=torch.device('cpu'))
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)

            self.normalization = CHECKPOINT_PATH_MAPPING[checkpoint_name]['norm']

        self.cutoff = CHECKPOINT_PATH_MAPPING[checkpoint_name]['default_cutoff']
        print('model load succeeds!')

    def load_default_cutoof(self, checkpoint_name, model_param={}):
        #TODO: merge into load_train
        self.cutoff = CHECKPOINT_PATH_MAPPING[checkpoint_name]['default_cutoff']

    def get_cutoff(self):
        return self.cutoff 

    def list_all_trained_models(self):
        for key, value in CHECKPOINT_PATH_MAPPING.items() :
            print(key)
    
    #def list_all_model_types(self):
    #    for key, value in MODEL_DEF_MAPPING.items() :
    #        print(key)

    def apply_on_single_zstack(self, input_img=None, filename=None, inputCh=None, normalization=None, already_normalized=False, cutoff=None, inference_param={}):

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        #torch.backends.cudnn.deterministic = False

        # check data
        if input_img is None:
            assert os.path.exists(filename)
            assert inputCh is not None 

            reader = AICSImage(filename)
            input_img = reader.data[0,[inputCh],:,:,:].astype(np.float32)
        else:
            input_img = input_img.astype(np.float32)
            # make sure the image has a dummy channel
            if not (len(input_img.shape)==4 and input_img.shape[0]==1):
                input_img = np.expand_dims(input_img, axis=0)

        if not already_normalized:
            args_norm = lambda:None
            if normalization is not None:
                args_norm.Normalization = normalization
            else:
                args_norm.Normalization = self.normalization
            
            input_img = input_normalization(input_img, args_norm)

        if 'ResizeRatio' in inference_param:
            ResizeRatio = inference_param['size_out']
            input_img = resize(input_img, (1, ResizeRatio[0], ResizeRatio[1], ResizeRatio[2]), method='cubic')
            for ch_idx in range(input_img.shape[0]):
                struct_img = input_img[ch_idx,:,:,:]
                struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                input_img[ch_idx,:,:,:] = struct_img
        else: 
            ResizeRatio = [1.0, 1.0, 1.0]

        # 
        model = self.model
        model.eval()

        # do padding on input
        padding = [(x-y)//2 for x,y in zip(self.size_in, self.size_out)]
        #img_pad = np.pad(input_img, ((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')
        img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'symmetric')#'constant')
        img_pad = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

        # we only support single output image in model zoo (other outputs are only supported in full segmenter prediction)
        assert len(self.OutputCh) ==2
        output_img = np.zeros(input_img.shape)

        # loop through the image patch by patch
        num_step_z = int(np.ceil(input_img.shape[1]/self.size_out[0]))
        num_step_y = int(np.ceil(input_img.shape[2]/self.size_out[1]))
        num_step_x = int(np.ceil(input_img.shape[3]/self.size_out[2]))
   
        with torch.no_grad():
            for ix in range(num_step_x):
                if ix<num_step_x-1:
                    xa = ix * self.size_out[2]
                else:
                    xa = input_img.shape[3] - self.size_out[2]

                for iy in range(num_step_y):
                    if iy<num_step_y-1:
                        ya = iy * self.size_out[1]
                    else:
                        ya = input_img.shape[2] - self.size_out[1]

                    for iz in range(num_step_z):
                        if iz<num_step_z-1:
                            za = iz * self.size_out[0]
                        else:
                            za = input_img.shape[1] - self.size_out[0]

                        input_patch = img_pad[ : ,za:(za+self.size_in[0]) ,ya:(ya+self.size_in[1]) ,xa:(xa+self.size_in[2])]
                        input_img_tensor = torch.from_numpy(input_patch)
                        tmp_out = model(Variable(input_img_tensor.cuda()).unsqueeze(0))
                        assert len(self.OutputCh)//2 <= len(tmp_out), print('the parameter OutputCh is not compatible with the number of output tensors')

                        label = tmp_out[self.OutputCh[0]]
                        prob = self.softmax(label)
                        out_flat_tensor = prob.cpu().data
                        out_tensor = out_flat_tensor.view(self.size_out[0], self.size_out[1], self.size_out[2], self.nclass[0])
                        out_nda = out_tensor.numpy()
                        output_img[0,za:(za+self.size_out[0]), ya:(ya+self.size_out[1]), xa:(xa+self.size_out[2])] = out_nda[:,:,:,self.OutputCh[1]]

        torch.cuda.empty_cache()

        if cutoff is None:
            # load default cutoff
            cutoff = self.cutoff 
        
        if cutoff>0:
            output_img = output_img>cutoff
            output_img = output_img.astype(np.uint8)
            output_img[output_img>0]=255
        else:
            output_img = (output_img - output_img.min()) / (output_img.max()-output_img.min())
            output_img = output_img.astype(np.float32)

        return output_img[0,:,:,:]


class SuperModel:

    def __init__(self, model_name, model_param={}):

        if not (model_name in SUPER_MODEL_MAPPING):
            raise IOError(f"model name '{model_name}' does not exist")

        
        self.model_list = SUPER_MODEL_MAPPING[model_name]['models']
        self.model_name = model_name
        self.models = []
            
        for mi, mname in enumerate(self.model_list):
            if mname[:2] == 'LF':
                model_def = parse_model(CHECKPOINT_PATH_MAPPING[mname]['path'])
                m = load_model_lf(model_def['path'], no_optim=True)
                m.to_gpu(0)
            else:
                m = SegModel()
                m.load_train(mname, model_param)
                m.load_default_cutoof(mname)
                m.to_gpu('cuda:0')
            
            self.models.append(m)
            
        print(SUPER_MODEL_MAPPING[model_name]['instruction'])

    
    
    def apply_on_single_zstack(self, input_img=None, filename=None, inputCh=None):

        # check data
        if input_img is None:
            assert os.path.exists(filename)
            assert inputCh is not None 

            reader = AICSImage(filename)
            input_img = reader.data.astype(np.float32)
            if input_img.shape[0]==1:
                input_img = np.squeeze(input_img,axis=0)
            elif input_img.shape[1]==1:
                input_img = np.squeeze(input_img,axis=1)
            else:
                print('error in data dimension')
                print(input_img.shape)
                quit()

            if input_img.shape[1]<input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))
            
            input_img = input_img[inputCh,:,:,:]
        else:
            input_img = input_img.astype(np.float32)
            # make sure the image has a dummy channel
            #if not (len(input_img.shape)==4 and input_img.shape[0]==1):
            #    input_img = np.expand_dims(input_img, axis=0)

        out = None
        if self.model_name == 'DNA_MEM_instance_production_alpha':
            from .DNA_MEM_instance_production_alpha import SegModule
            out_1, out_2, seed_out = SegModule(input_img, self.models)
            return out_1, out_2, seed_out
        elif self.model_name == 'LMNB1_morphological_production_alpha':
            from .LMNB1_morphological_production_alpha import SegModule
            return SegModule(input_img, self.models)
        elif self.model_name == 'LMNB1_morphological_RnD':
            from .LMNB1_morphological_production_alpha import SegModule
            seg, seg_name = SegModule(input_img, self.models, output_type='RnD')
            #seg, seg_name, pred, pred_name = SegModule(input_img, self.models, return_prediction=True)
            return seg, seg_name
        elif self.model_name == 'LMNB1_morphological_with_labelfree':
            from .LMNB1_morphological_with_labelfree import SegModule
            seg, seg_name = SegModule(input_img, self.models)
            return seg, seg_name
        elif self.model_name == 'LMNB1_fill_instance_100x_hipsc':
            from .LMNB1_fill_instance_100x_hipsc import SegModule
            seg, seg_name = SegModule(input_img, self.models, return_prediction=False)
            return seg, seg_name
        elif self.model_name == 'H2B_instance_hipsc_ZSD_40x':
            return None      
        elif self.model_name == 'Nucleus_instance_hipsc_880' or self.model_name == 'Nucleus_instance_hipsc_ZSD' or self.model_name == 'Nucleus_instance_cardio_ZSD' or self.model_name =='Nucleus_instance_cardio_880':
            from .Nucleus_instance_test import SegModule
            out = SegModule(input_img, self.models, model_name=self.model_name)
            return out
        elif self.model_name == 'DNA_instance_from_labelfree_alpha':
            from .DNA_instance_lf import SegModule
            out = SegModule(input_img, self.models, model_name=self.model_name)
            return out
        elif self.model_name == 'LF_DNA_instance_alpha':
            from .LabelFree_DNA_instance import SegModule
            out = SegModule(input_img, self.models)
            return out
        elif self.model_name == 'DNA_MEM_instance_LF_integration' or self.model_name == 'DNA_MEM_instance_LF_integration_two_camera':
            from .DNA_MEM_instance_plus_LF import SegModule
            out = SegModule(input_img, self.models, two_camera=False)
            return out
        elif self.model_name == 'DNA_MEM_instance_CAAX_with_BF':
            from .DNA_MEM_instance_caax_with_BF import SegModule
            out = SegModule(input_img, self.models, two_camera=False)
            return out
        elif self.model_name == 'DNA_instance_LF_integration_two_camera':
            from .DNA_instance_plus_LF import SegModule
            out_1, out_2 = SegModule(input_img, self.models, two_camera=True)
            return out_1, out_2
        elif self.model_name == 'structure_H2B_production':
            from .structure_H2B_100x_hipsc import SegModule
            return SegModule(input_img, self.models)
        elif self.model_name == 'structure_AAVS1_production':
            from .structure_AAVS1_100x_hipsc import SegModule
            return SegModule(input_img, self.models) 


def list_all_super_models():
    import sys 
    if len(sys.argv)==2:
        print(SUPER_MODEL_MAPPING[sys.argv[1]]['instruction'])
    elif len(sys.argv)==1:
        for key, value in SUPER_MODEL_MAPPING.items() :
            print(key)
            #print(value['instruction'])
    else:
        print('error function')
        quit()
