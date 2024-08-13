# AIDI_1002_FINAL_PROJECT

### Title:End-to-End Object Detection with Transformers
#### Group Member Names : 
Sanjana Manish Modi

Dayana Singh

# Implement paper code :
*********************************************************************************************************************

### Usage - Object detection
a.
First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git 
or 
git clone https://github.com/Sanjanamodi/AIDI_1002_FINAL_PROJECT.git
and navigate to 'Research-paper-code'
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://coThg/#download).
We expeshould the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
A single epoch takes 28 minutes, so 300 epoch training
takes around 6 dainTmachine), achieving 39.5/60.3 AP/AP50.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and he whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```

*********************************************************************************************************************
### Contribution  Code :

In the Colab notebook, We've configured GPU support. Initially, we explored a pre-trained model and then extended it by incorporating a custom dataset focused on fruits. This extended model was subsequently trained using the new dataset. The fruit dataset is structured similarly to the COCO dataset and was sourced from Roboflow. Due to limitations with GPU availability, the model was trained for only 3 epochs.

Notebook link: [Final_Assignment_MLP.ipynb](https://github.com/Sanjanamodi/AIDI_1002_FINAL_PROJECT/blob/main/Contribution-code/Final_Assignment_MLP.ipynb)

> [!IMPORTANT]
> Please use notebook to run the contribution code with GPU.

### Library used for project
- [x] pytorch
- [x] torchvision
- [x] pycocotools
- [x] cython scipy
- [x] scipy
- [x] transformers
- [x] pytorch-lightning
- [x] roboflow
- [x] timm
- [x] supervision
- [x] random
- [x] cv2
- [x] numpy
- [x] coco_eval