# Introduction
[DeepMask](http://arxiv.org/abs/1506.06204) is a category agnostic object proposal algorithm based on a discriminative convolutional neural network. It can be used in any vision algorithm requiring either box or segmentation object proposals. [SharpMask](http://arxiv.org/abs/1603.08695) is a extension of DeepMask which generate more refined masks using a top-down refinement step.

This is a Torch implementation of DeepMask and SharpMask. This repository contains the code to train and evaluate both DeepMask and SharpMask on [MSCOCO](http://mscoco.org) dataset.

If you use DeepMask/SharpMask in your research, we appreciate it if you cite the appropriate papers:
```
@inproceedings{PinheiroNIPS15deepmask,
   title = {Learning to Segment Object Candidates},
   author = {Pinheiro, Pedro O and Collobert, Ronan and Dollar, Piotr},
   booktitle = {NIPS},
   year = {2015}
}
```
```
@article{PinheiroArXiv16sharpmask,
   title = {Learning to Refine Object Segments},
   author = {Pinheiro, Pedro O and Lin, Tsung-Yi and Collobert, Ronan and Dollar, Piotr},
   journal = {arXiv},
   year = {2016}
}
```
## Requirements
DeepMask requires or work with
* MAC OS X or Linux
* NVIDIA GPU with compute capability of 3.5 or above.

## Installing Dependencies
* Install/update [Torch](http://torch.ch)
* Make sure the following packages are installed: [image](https://github.com/torch/image), [tds](https://github.com/torch/tds), [cjson](https://github.com/clementfarabet/lua---json), [nnx](https://github.com/clementfarabet/lua---nnx), [optim](https://github.com/torch/optim), [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch), [COCO API](https://github.com/pdollar/coco)

## Quick Start
To use DeepMask/SharpMask as an off-the-shelf object proposal generator is very easy:

1. Download DeepMask code
   ```bash
   git clone https://github.com/facebookresearch/deepmask
   cd deepmask
   ```
We will refer to cloned directory as `$PATH_DEEPMASK`.

2. Download pre-trained DeepMask/SharpMask models. We assume the pre-trained models is located in `$PATH_DEEPMASK/pretrained/`
   ```bash
   mkdir -p $PATH_DEEPMASK/pretrained; cd $PATH_DEEPMASK/pretrained
   mkdir -p $PATH_DEEPMASK/pretrained/deepmask; mkdir -p $PATH_DEEPMASK/pretrained/sharpmask
   cd $PATH_DEEPMASK/pretrained/deepmask; wget http://LINK_TO_DEEPMASK # for pretrained DeepMask
   cd ../sharpmask;  wget http://LINK_TO_SHARPMASK # for pretrained SharpMask
   ```

3. Run `computeProposals.lua`. This script requires as argument the path to a pretrained model. The optional argument `-img` gives the path to a sample image.
   ```bash
   # compute on a sample image provided in this repository
   th computeProposals.lua $PATH_DEEPMASK/pretrained/deepmask # use pretrained deepmask
   th computeProposals.lua $PATH_DEEPMASK/pretrained/sharpmask # use pretrained sharpmask

   # provide a path to a test image
   th computeProposals.lua $PATH_DEEPMASK/sharpmask -img /path/to/test/image
   ```

## Training Your Own Model
To train your own DeepMask/SharpMask model, follow these steps:

### Preparation
1. Download DeepMask code
   ```bash
   git clone PATH_TO_DEEPMASK_REPOSITORY
   cd deepmask
   ```
We will refer to cloned directory as `$PATH_DEEPMASK`.

2. Download Torch [ResNet-50](LINK_TO_RESNET-50.t7) model pretrained on Imagenet. We assume the model is located in `$PATH_DEEPMASK/pretrained`.
   ```bash
   mkdir -p $PATH_DEEPMASK/pretrained
   cd $PATH_DEEPMASK/pretrained
   wget LINK_TO_RESNET-50.t7
   ```

3. Download [MSCOCO](http://mscoco.org) images and annotations. We assume the data is located in `$PATH_DEEPMASK/data`.
   ```bash
   mkdir -p $PATH_DEEPMASK/data
   cd $PATH_DEEPMASK/data
   wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
   wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
   wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
   ```

### Training
To train your model, you need to launch the `train.lua` script. It contains several options. To list them, simply use the `--help` flag.

1. To train DeepMask:
   ```bash
   th train.lua
   ```

2. To train Sharpmask:
   ```bash
   th train.lua -dm path/to/trained/deepmask/
   ```
The second option requires a trained deepmask file. You can either train your own or use a pre-trained DeepMask.

### Evaluation
There are two ways to evaluate a model on the MSCOCO datasaet.

1. `evalPerPatch.lua` evaluates only the mask generation, assuming each image contains an object located in the center of the input patch:
   ```bash
   th evalPerPatch.lua path/to/trained/deepmask-or-sharpmask/
   ```

2. `evalPerImage.lua` evaluates the model in full-scene scenario on MSCOCO dataset, as described in the paper. By default, it evaluate the first 5K images of MSCOCO dataset (run `th evalPerImage.lua --help` to see the options):
   ```bash
   th evalPerImage.lua path/to/trained/deepmask-or-sharpmask/
   ```

## Precomputed Proposals
Here you can download pre-computed 1000 proposals to MSCOCO and PASCAL VOC 2007/2012 datasets, for both segm and bbox proposals. The proposals are divided in chunks of 500 images each (that is, each json contain 1000 proposals of 500 images).

### Segmentation Proposals
* PASCAL VOC 2007 [train/val/test sets](LINK_TO_VOC07_SEGM)
* PASCAL VOC 2012 [train/val/test sets](LINK_TO_VOC12_SEGM)
* MSCOCO  [train set](LINK_TO_COCO_TRAIN_SEGM)
* MSCOCO  [val set](LINK_TO_COCO_VAL_SEGM)
* MSCOCO  [test-dev set](LINK_TO_COCO_TEST-DEV_SEGM)
* MSCOCO  [test-full set](LINK_TO_COCO_TEST-FULL_SEGM)

### Bounding Box Proposals
* PASCAL VOC 2007 [train/val/test sets](LINK_TO_VOC07_BBOX)
* PASCAL VOC 2012 [train/val/test sets](LINK_TO_VOC12_BBOX)
* MSCOCO  [train set](LINK_TO_COCO_TRAIN_BBOX)
* MSCOCO  [val set](LINK_TO_COCO_VAL_BBOX)
* MSCOCO  [test-dev set](LINK_TO_COCO_TEST-DEV_BBOX)
* MSCOCO  [test-full set](LINK_TO_COCO_TEST-FULL_BBOX)
