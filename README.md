This is the official implementation for the paper [Improving Confidence Estimates for Unfamiliar Examples](https://arxiv.org/abs/1804.03166) presented in CVPR 2020. See also the [project page](https://zhizhongli.vision/projects/improving-confidence-estimates).

This repository currently provides evaluation of models on familiar and unfamiliar splits of the four datasets using the various metrics, as described in the paper. Baseline models and calibration parameters are also available.

Training code with hyperparameters will be made available some time in the future.

# Install
Please use the following code for a conda installation of the dependencies with versions we tested on. This repository is made compatible with the newest PyTorch version v1.5.1. (It was originally developed under PyTorch r0.1.11)

```
conda create --name confeval \
    python=3 scipy=1.5.0 numpy=1.18.5 scikit-learn=0.23.1 h5py=2.10.0 \
    pycocotools=2.0.1 tqdm=4.46.1 easydict=1.9 pytorch=1.5.1 \
    torchvision=0.6.1 cudatoolkit=10.2 \
    -c pytorch -c conda-forge -y
conda activate confeval
```

Please download datasets that you need (ImageNet, VOC 2012, COCO, Pets, LFW+) from respective sources. Please modify dataset paths in `conf_eval/paths.py` and make sure that the directory structure has at least these contents:

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
Some subfolders may need some moving around, especially VOC and MSCOCO.

# Usage: evaluation
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

Modify the `lookup` dictionary or use the `eval_dset` function to suit your own needs. 
The evaluation criteria are implemented in the `SafeProbsMC` and `SafeProbsML` classes in `conf_eval/utils.py`.

# Usage: training
Coming soon: code for baseline training and calibration. Implementation of other methods in the paper is available upon request.

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
