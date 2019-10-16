# VQA Baseline
Baseline VQA PyTorch implementation for Open-Ended Question-Answering



---
## Table of Contents

> The project comprises of the following sections.
- [Dataset Preparation](#dataset-preparation)
- [Dataset Format](#dataset-format)
- [Architecture](#architecture)
- [Training](#training)
- [References](#references)

---

## Dataset Preparation

We use images from the MS COCO dataset

- The images are partitioned into two sets of object categories, with varying degrees of difficulty: <br>
  <b>...</b> <br>
  Set A 
  * ...
  
  
- <b><i>TO-DO</i></b>:: Instead of using the entire cropped regions as candidates, mask out the background 
  within the bounding boxes using the corresponding segmentation masks. The original context image however, 
  will have the box region masks <i>(to avoid silhouettes in the context image) </i>. <br>
  This could be helpful in avoiding the model from learning spurious correlations between the background of 
  candidate regions & the context image.

Given the two-set category partition file, extracts the context image along with four candidate images
and saves to disk.

```bash
$ python3 prepare_data_coco.py --inp_dir /home/axe/Datasets/MS_COCO/coco \
--partition_file /home/axe/Projects/MCS_VL/dataset_category_partitions.txt \
--coco_v train2017 --out_dir /home/axe/Datasets/Scene_Context_Dataset
```

This generates the dataset in the following format: <br>
- $out_dir/images : consists of query & candidate images (where the first candidate is the ground-truth)
- $out_dir/img_context_candidates.txt: <br>`context_img cand_1_img cand_2_img cand_3_img cand_4_img`  

---
## Dataset Format

Once we have adapted the COCO dataset to query-candidates format, we run the following script to 
convert to <b>triplets</b> (anchor, positive, negative)  <br>
The train-validation split is performed here.

```bash
$ python3 prepare_data_final.py --inp_file_path TO-DO --prob_split TO-DO 
```

The final dataset file has the following format:  <br>
`context_anchor_img  positive_img  negative_img`

---
## Architecture

The architecture can be summarized as:-

Context Img --> vgg11_head --> ctx_encoder --> ctx_embedding <br>
Candidate Img --> vgg11_head --> cand_encoder --> cand_embedding 

```
ContextNet(
  # Common Feature Extractor (trainable)
  (vgg11_head): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  
  # Conv & Subsample Context 
  (ctx_encoder): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  
  # Conv & Subsample Candidate
  (cand_encoder): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  
  # Final 1D Embeddings
  (ctx_embedding): Linear(in_features=25088, out_features=1024, bias=True)
  (cand_embedding): Linear(in_features=25088, out_features=1024, bias=True)
)

```

---

## Training

Run the following script for training:

```bash
 python3 main.py --mode train --model_name demo_model \
 --root_dir /home/axe/Datasets/Scene_Context_Dataset --log_dir /home/axe/Projects/MCS_VL/results_log/ \
 --train_file img_triplets_train.txt --val_file img_triplets_val.txt --batch_size 2 --num_epochs 10
```


> *TODO* : Add TensorBoardX support


- [ ] TensorBoardX
- [ ] Online Triplet Mining

---

## References
[1]  [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379)
