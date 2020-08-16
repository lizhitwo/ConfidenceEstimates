This is the official implementation for the paper [Improving Confidence Estimates for Unfamiliar Examples](https://arxiv.org/abs/1804.03166) presented in CVPR 2020. See also the [project page](https://zhizhongli.vision/projects/improving-confidence-estimates).

This repository provides evaluation of models on familiar and unfamiliar splits of the four datasets using the various metrics, as described in the paper. Baseline models and calibration parameters are also available.

**Update:** Training code with hyperparameters is also available for single models, ensemble models, and performing calibration with the validation set.

# Install
1. Please use the following code for a conda installation of the dependencies with versions we tested on. This repository is made compatible with the newest PyTorch version v1.5.1. (It was originally developed under PyTorch r0.1.11)
```
conda create --name confeval \
    python=3 scipy=1.5.0 numpy=1.18.5 scikit-learn=0.23.1 h5py=2.10.0 \
    pycocotools=2.0.1 tqdm=4.46.1 easydict=1.9 pytorch=1.5.1 \
    torchvision=0.6.1 cudatoolkit=10.2 pytorch-lightning=0.8.4 \
    -c pytorch -c conda-forge -y
conda activate confeval
```
2. Please download datasets that you need (ImageNet, VOC 2012, COCO, [Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [LFW+](http://biometrics.cse.msu.edu/Publications/Databases/MSU_LFW+/)) from respective sources. 
3. Please modify dataset paths in `conf_eval/paths.py` and make sure that the directory structure has at least these contents:
```
$ ls path/to/root/{ILSVRC2012,VOC2012,MSCOCO,Oxford_IIIT_Pets,LFW+_Release} -p

path/to/root/ILSVRC2012:
ILSVRC2012_devkit_t12/  ILSVRC2012_img_test/  ILSVRC2012_img_train/  ILSVRC2012_img_val/

path/to/root/LFW+_Release:
lfw+_jpg24/  lfw+_labels.mat

path/to/root/MSCOCO:
annotations/  test2015/  train2014/  val2014/

path/to/root/Oxford_IIIT_Pets:
annotations/  images/

path/to/root/VOC2012:
ImageSets/  JPEGImages/
```
4. Some subfolders may need some moving around, especially VOC and MSCOCO.
5. Finally, please copy `put_in_Pets_annotations/train.txt` and `put_in_Pets_annotations/val.txt` into `path/to/root/Oxford_IIIT_Pets/annotations/`. 


# Usage: evaluation
### For released models
Use these commands to evaluate the released baseline models. 
```
python evaluate.py lfwp_gender
python evaluate.py cat_vs_dog
python evaluate.py imnet_animal
python evaluate.py voc_to_coco
```
Use these commands to evaluate them with temperature scaling calibration.
```
python evaluate.py lfwp_gender  --calibrate
python evaluate.py cat_vs_dog   --calibrate
python evaluate.py imnet_animal --calibrate
python evaluate.py voc_to_coco  --calibrate
```
The resulting values will be slightly different from the paper tables, because the tables are the average results of 10 runs for each experiment.

### For your own models
The evaluation criteria are implemented in the `SafeProbsMC` and `SafeProbsML` classes in `conf_eval/utils.py`.
```
# For multi-class softmax logits (NxC shape) and enumerated ground truth (N shape):
prob = SafeProbsMC.from_logits(logits)
print(prob.dict_performance(gt))

# For multi-label sigmoid logits (NxC shape) and multi-label binary ground truth (NxC shape):
prob = SafeProbsML.from_logits(logits)
print(prob.dict_performance(gt))
```

Alternatively, copy and modify the `lookup` dictionary or use the `eval_dset` function to suit your own needs. 

# Usage: training
We provide code for training single or ensemble models and calibration with temperature scaling. Implementation of other methods in the paper is available upon request.

Please make sure you have downloaded the [PyTorch Places365-pretrained models](https://github.com/CSAILVision/places365) to the `pretrained` folder under your dataset root:
```
$ ls path/to/root/pretrained -p
densenet161_places365.pth.tar  resnet18_places365.pth.tar
```
The installation section has changed since the evaluation code release. Please double check that you have installed Pytorch-Lightning and copied the train/val split of the Pets dataset.

Use these commands to train corresponding models:
```
# Set dataset name. Choices: lfwp_gender cat_vs_dog imnet_animal voc_to_coco
DSET=lfwp_gender

# Single model
python train.py ${DSET} test
# Single model (average performance over 10 runs)
python train.py ${DSET} test --i_runs {0..9}

# Single model with T-scaling calibration
# Calibrate on validation split
python train.py ${DSET} val --get_calibration
# Run with temperature obtained above
python train.py ${DSET} test --calibrate <temperature reported above>
# Run with temperature obtained above (average performance over 10 runs)
python train.py ${DSET} test --calibrate <temperature reported above> --i_runs {0..9}

# Ensemble of 10 models from the 10 runs
python train.py ${DSET} test --ensemble --i_runs {0..9}

# Ensemble of 10 models from the 10 runs with T-scaling calibration
python train.py ${DSET} test --ensemble --calibrate <temperature reported above> --i_runs {0..9}
```
The resulting values will be slightly different from the paper tables and provided calibration temperatures due to randomness in deep learning.

# Citation
If you use this repository or find any component useful, please consider citing our paper:

```
@inproceedings{li2020improving,
  title={Improving Confidence Estimates for Unfamiliar Examples},
  author={Li, Zhizhong and Hoiem, Derek},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2686--2695},
  year={2020}
}
```

# LICENSE
This repository is released under the MIT license. Please see the `LICENSE` file for the license text.
