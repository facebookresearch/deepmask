# Introduction
This repository contains a [Torch](http://torch.ch) implementation for both the [DeepMask](http://arxiv.org/abs/1506.06204) and [SharpMask](http://arxiv.org/abs/1603.08695) object proposal algorithms.

![teaser](https://raw.githubusercontent.com/facebookresearch/deepmask/master/data/teaser.png)

[DeepMask](http://arxiv.org/abs/1506.06204) is trained with two objectives: given an image patch, one branch of the model outputs a class-agnostic segmentation mask, while the other branch outputs how likely the patch is to contain an object. At test time, DeepMask is applied densely to an image and generates a set of object masks, each with a corresponding objectness score. These masks densely cover the objects in an image and can be used as a first step for object detection and other tasks in computer vision.

[SharpMask](http://arxiv.org/abs/1603.08695) is an extension of DeepMask which generates higher-fidelity masks using an additional top-down refinement step. The idea is to first generate a coarse mask encoding in a feedforward pass, then refine this mask encoding in a top-down pass using features at successively lower layers. This result in masks that better adhere to object boundaries.

If you use DeepMask/SharpMask in your research, please cite the relevant papers:
```
@inproceedings{DeepMask,
   title = {Learning to Segment Object Candidates},
   author = {Pedro O. Pinheiro and Ronan Collobert and Piotr Dollár},
   booktitle = {NIPS},
   year = {2015}
}
```
```
@inproceedings{SharpMask,
   title = {Learning to Refine Object Segments},
   author = {Pedro O. Pinheiro and Tsung-Yi Lin and Ronan Collobert and Piotr Dollár},
   booktitle = {ECCV},
   year = {2016}
}
```
Note: the version of DeepMask implemented here is the updated version reported in the SharpMask paper. DeepMask takes on average .5s per COCO image, SharpMask runs at .8s. Runtime roughly doubles for the "zoom" versions of the models.

# Requirements and Dependencies
* MAC OS X or Linux
* NVIDIA GPU with compute capability 3.5+
* [Torch](http://torch.ch) with packages: [COCO API](https://github.com/pdollar/coco), [image](https://github.com/torch/image), [tds](https://github.com/torch/tds), [cjson](https://github.com/clementfarabet/lua---json), [nnx](https://github.com/clementfarabet/lua---nnx), [optim](https://github.com/torch/optim), [inn](https://github.com/szagoruyko/imagine-nn), [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch)

# Quick Start
To run pretrained DeepMask/SharpMask models to generate object proposals, follow these steps:

1. Clone this repository into $DEEPMASK:

   ```bash
   DEEPMASK=/desired/absolute/path/to/deepmask/ # set absolute path as desired
   git clone git@github.com:facebookresearch/deepmask.git $DEEPMASK
   ```

2. Download pre-trained DeepMask and SharpMask models:

   ```bash
   mkdir -p $DEEPMASK/pretrained/deepmask; cd $DEEPMASK/pretrained/deepmask
   wget https://s3.amazonaws.com/deepmask/models/deepmask/model.t7
   mkdir -p $DEEPMASK/pretrained/sharpmask; cd $DEEPMASK/pretrained/sharpmask
   wget https://s3.amazonaws.com/deepmask/models/sharpmask/model.t7
   ```

3. Run `computeProposals.lua` with a given model and optional target image (specified via the `-img` option):

   ```bash
   # apply to a default sample image (data/testImage.jpg)
   cd $DEEPMASK
   th computeProposals.lua $DEEPMASK/pretrained/deepmask # run DeepMask
   th computeProposals.lua $DEEPMASK/pretrained/sharpmask # run SharpMask
   th computeProposals.lua $DEEPMASK/pretrained/sharpmask -img /path/to/image.jpg
   ```


# Training Your Own Model
To train your own DeepMask/SharpMask models, follow these steps:

## Preparation
1. If you have not done so already, clone this repository into $DEEPMASK:

   ```bash
   DEEPMASK=/desired/absolute/path/to/deepmask/ # set absolute path as desired
   git clone git@github.com:facebookresearch/deepmask.git $DEEPMASK
   ```

2. Download the Torch [ResNet-50](https://s3.amazonaws.com/deepmask/models/resnet-50.t7) model pretrained on ImageNet:

   ```bash
   mkdir -p $DEEPMASK/pretrained; cd $DEEPMASK/pretrained
   wget https://s3.amazonaws.com/deepmask/models/resnet-50.t7
   ```

3. Download and extract the [COCO](http://mscoco.org/) images and annotations:

   ```bash
   mkdir -p $DEEPMASK/data; cd $DEEPMASK/data
   wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
   wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
   wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
   ```

## Training
To train, launch the `train.lua` script. It contains several options, to list them, simply use the `--help` flag.

1. To train DeepMask:

   ```bash
   th train.lua
   ```

2. To train SharpMask (requires pre-trained DeepMask model):

   ```bash
   th train.lua -dm /path/to/trained/deepmask/
   ```

## Evaluation
There are two ways to evaluate a model on the COCO dataset.

1. `evalPerPatch.lua` evaluates only the mask generation step. The per-patch evaluation only uses image patches that contain roughly centered objects. Its usage is as follows:

   ```bash
   th evalPerPatch.lua /path/to/trained/deepmask-or-sharpmask/
   ```

2. `evalPerImage.lua` evaluates the full model on COCO images, as reported in the papers. By default, it evaluates performance on the first 5K COCO validation images (run `th evalPerImage.lua --help` to see the options):

   ```bash
   th evalPerImage.lua /path/to/trained/deepmask-or-sharpmask/
   ```


# Precomputed Proposals

You can download pre-computed proposals (1000 per image) on the COCO and PASCAL VOC datasets, for both segmentation and bounding box proposals. We use the COCO JSON [format](http://mscoco.org/dataset/#format) for the proposals. The proposals are divided into chunks of 500 images each (that is, each JSON contains 1000 proposals per image for 500 images). All proposals correspond to the "zoom" setting in the paper (DeepMaskZoom and SharpMaskZoom) which tend to be most effective for object detection.

## DeepMask
* COCO Boxes: [[train](https://s3.amazonaws.com/deepmask/boxes/deepmask-coco-train-bbox.tar.gz) | [val](https://s3.amazonaws.com/deepmask/boxes/deepmask-coco-val-bbox.tar.gz) | [test-dev](https://s3.amazonaws.com/deepmask/boxes/deepmask-coco-test-dev-bbox.tar.gz) | [test-full](https://s3.amazonaws.com/deepmask/boxes/deepmask-coco-test-full-bbox.tar.gz)]
* COCO Segments: [[train](https://s3.amazonaws.com/deepmask/segms/deepmask-coco-train.tar.gz) | [val](https://s3.amazonaws.com/deepmask/segms/deepmask-coco-val.tar.gz) | [test-dev](https://s3.amazonaws.com/deepmask/segms/deepmask-coco-test-dev.tar.gz) | [test-full](https://s3.amazonaws.com/deepmask/segms/deepmask-coco-test-full.tar.gz)]
* PASCAL Boxes: [[train+val+test-2007](https://s3.amazonaws.com/deepmask/boxes/deepmask-pascal07-bbox.tar.gz) | [train+val+test-2012](https://s3.amazonaws.com/deepmask/boxes/deepmask-pascal12-bbox.tar.gz)]
* PASCAL Segments: [[train+val+test-2007](https://s3.amazonaws.com/deepmask/segms/deepmask-pascal07.tar.gz) | [train+val+test-2012](https://s3.amazonaws.com/deepmask/segms/deepmask-pascal12.tar.gz)]

## SharpMask
* COCO Boxes: [[train](https://s3.amazonaws.com/deepmask/boxes/sharpmask-coco-train-bbox.tar.gz) | [val](https://s3.amazonaws.com/deepmask/boxes/sharpmask-coco-val-bbox.tar.gz) | [test-dev](https://s3.amazonaws.com/deepmask/boxes/sharpmask-coco-test-dev-bbox.tar.gz) | [test-full](https://s3.amazonaws.com/deepmask/boxes/sharpmask-coco-test-full-bbox.tar.gz)]
* COCO Segments: [[train](https://s3.amazonaws.com/deepmask/segms/sharpmask-coco-train.tar.gz) | [val](https://s3.amazonaws.com/deepmask/segms/sharpmask-coco-val.tar.gz) | [test-dev](https://s3.amazonaws.com/deepmask/segms/sharpmask-coco-test-dev.tar.gz) | [test-full](https://s3.amazonaws.com/deepmask/segms/sharpmask-coco-test-full.tar.gz)]
* PASCAL Boxes: [[train+val+test-2007](https://s3.amazonaws.com/deepmask/boxes/sharpmask-pascal07-bbox.tar.gz) | [train+val+test-2012](https://s3.amazonaws.com/deepmask/boxes/sharpmask-pascal12-bbox.tar.gz)]
* PASCAL Segments: [[train+val+test-2007](https://s3.amazonaws.com/deepmask/segms/sharpmask-pascal07.tar.gz) | [train+val+test-2012](https://s3.amazonaws.com/deepmask/segms/sharpmask-pascal12.tar.gz)]
