# segmenter_model_zoo

[![Build Status](https://github.com/AllenCell/segmenter_model_zoo/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/segmenter_model_zoo/actions)
[![Documentation](https://github.com/AllenCell/segmenter_model_zoo/workflows/Documentation/badge.svg)](https://AllenCell.github.io/segmenter_model_zoo)
[![Code Coverage](https://codecov.io/gh/AllenCell/segmenter_model_zoo/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCell/segmenter_model_zoo)

Batch processing script and functions for running deep learning based segmentation models released by Allen Institute for Cell Science. 

---

## Features
* Reproducing deep learning based segmentation results in Allen Cell Data Collection
* Pulling models from Allen Cell quilt buckets automatically
* Supporting both command line batch processing and python API callable in other functions
* Serving as a starting point for iterative deep learning on your own data
* Adding more models continously when ready

## System requirement

Most of the models can run at any Linux or Windows machine with NVIDIA GPU properly
setup up. The only exception is the full version of cells and nuclei segmentation used
in Allen Cell Data Collection, which uses an object detection model that requires 
compiling certain CUDA code in Linux. So the full version is only supported in Linux, but 
we provide a simplified version without object detection that can run on both Windows and Linux, which will also be useful for most users. See more details in the list of models below.

Note: If you are using Windows and see the an error like `error: Microsoft Visual C++ 14.0 or greater is required. Get it with Microsoft C++ Build Tools`, then juse follow the detailed intruction in the error message to download Microsoft Build Tools. The specific version we used is *MSVC v142 - VS 2019 C++ x64/x86 build tools (v14.27)*

## Installation

### Step 1: Install PyTorch from official website

See [official instruction](https://pytorch.org/get-started/locally/). We have tested on 1.3.1, 1.4.0, 1.6.0, 1.7.0 on CUDA 10.1 and CUDA 10.2

### Step 2: Install segmenter_model_zoo

**Install stable release and use as it:** 
`pip install segmenter_model_zoo`<br>

**Install nightly development head and make customization:**

```bash
git clone git@github.com:AllenCell/segmenter_model_zoo.git
cd segmenter_model_zoo
pip install -e .[all]
```

### Step 3 (Optional): Install pytorch-fnet and cell-detector if you want to use the full version of cells and nuclei segmentation (for Linux only)

**Install pytorch-fnet for running label-free model**
```bash
git clone https://github.com/AllenCellModeling/pytorch_fnet.git
cd pytorch_fnet
git checkout ad_customization
pip install .
```

**Install cell-detector**

```bash
git clone ssh://git@aicsbitbucket.corp.alleninstitute.org:7999/assay/cell_detector.git
cd cell_detector
pip install .
cd ./cell_detector/bin/apex
python setup.py install --cuda_ext --cpp_ext
cd ..  # assume you are under /cell_detector/cell_detector/bin
rm -rf build
python setup.py build develop
```

Note: Depending on your CUDA version and different generations of your GPU, you may need different pytorch version to make cell-detector working properly. We found `pytorch==1.3.1` and `torchvision==0.4.2` is a stable version for both CUDA 10.1 and CUDA 10.2 and working well for NVDIA titanx, titanxp, gtx1080, v100 GPUs. So, if you find installation failed on your current environment, you may switch to `pytorch==1.3.1` and `torchvision==0.4.2`.

## Documentation

In general there two types of models: basic models and super models. Basic model is just a trained neural network. Super model can be multiple basic models + certain preprocesisng and post processing steps to generate the final output.

Current basic models:

* [v] DNA_mask_production: from nuclei images (via DNA dye) to the semantic segmentation mask of DNA
* [v] DNA_seed_production: from nuclei images (via DNA dye) to genearate one seed per DNA
* [v] CellMask_edge_production: from cell membrane images (via CellMask dye) to semantic cell membrane segmentation
* [v] CAAX_production: from CAAX images to semantic CAAX segmentation
* [v] LMNB1_all_production: from lamin B1 images to the semanic lamin b1 segmentation
* [v] LMNB1_fill_production: from lamin B1 images to the mask of nuclei (filled shapes) except mitotic cells
* [v] LMNB1_seed_production: from lamin B1 images to generate one seed per nucleus
* [v] H2B_coarse: from H2B images to coarse H2B segmentation. NOTE: this is not the H2B segmentation provided in the 
      Allen Cell Data Collection. This is a coarse version, which can be roughly considered as equivelant to nuclear
      segmentation

Current super models:

* [v] DNA_MEM_instance_plus_LF: full version of cells and nuclei segmentaiton for images in Allen Cell Data Collection that 
      were acquired with old single camera pipeline
* [v] DNA_MEM_instance_plus_LF_two_camera: full version of cells and nuclei segmentaiton for images in Allen Cell Data Collection that 
      were acquired with current two-camera pipeline
* [v] DNA_MEM_instance_basic: simplified version of cells and nuclei segmentaiton applicable for all images in Allen Cell Data 
      Collection. Comparing to the full version, labelfree and cell pair correction were excluded for simplicity.
* [v] LMNB1_morphological_production_alpha: lamin B1 morphological segmentation for the lamib B1 cell line in Allen Cell Data 
      Collection. Comparing to basic semantic lamib B1 segmentation, the morphological segmentation refers that the lamin shells are fully closed 
      (i.e. topologically fillable in 3D)
* [v] structure_AAVS1_100x_hipsc: from CAAX images for final binary results of semantic CAAX segmentation
* [v] structure_H2B_100x_hipsc: from H2B images for final binary results of coarse H2B segmentation

There are a few auxillary models available as part of the full version of cells and nuclei segmentation
* [v] LF_DNA_mask: label-free model predicting DNA segmentation mask from bright field
* [v] LF_DNA_mask_two_camera: label-free model predicting DNA segmentation mask from bright field, but only for images acquired uisng two-camera pipeline (slightly different bright field images)
* [v] LF_mem_edge: label-free model predicting cell membrane segmentation from bright field
* [v] LF_mem_edge_two_camera: label-free model predicting cell membrane segmentation from bright field, but only for images acquired uisng two-camera pipeline (slightly different bright field images)
* [v] cell_pair_det_prod: a 2D object detection model to identify the pair of daughter cells during late mitosis


*NOTE: the terms used here, e.g. nuclei images, cell membrane images, two-camera pipeline and single camera pipeline, etc., omit more bioloigical and imaging details which could be important to the success of applying our models on images not from Allen Cell Data Collection. Please refer Allen Cell Data Collection for details (e.g., the imaging modality or the z step size could be important).*

To use these models:

(1) You can prepare a yaml configuration file (see examples [HERE](https://github.com/AllenCell/segmenter_model_zoo/tree/main/config_examples), just specify which super model to use and input/output path) and then run
```bash
run_model_zoo --config /path/to/your/config.yaml --debug
```

(2) You can run basic models or super models using provided python API. See the demo jupyter notebooks [HERE](https://github.com/AllenCell/segmenter_model_zoo/tree/main/playbooks).

For full package documentation please visit [AllenCell.github.io/segmenter_model_zoo](https://AllenCell.github.io/segmenter_model_zoo).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Questions?

If you have any questions, feel free to leave a comment in our Allen Cell forum: [https://forum.allencell.org/](https://forum.allencell.org/). 

***Free software: Allen Institute Software License***

