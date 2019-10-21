# VQA Baseline
Baseline VQA PyTorch implementation for Open-Ended Question-Answering



---
## Table of Contents

> The project comprises of the following sections.
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training](#training)
- [References](#references)

---

## Dataset

Given the <a href="https://visualqa.org/download.html">VQA Dataset's</a> 
annotations & questions file, generates a dataset file (.txt) in the following format:

`image_name` \t `question` \t `answer`

- image_name is the image file name from the COCO dataset <br>
- question is a comma-separated sequence <br>
- answer is a string (label) <br>

Sample Execution:

```bash
$ python3 prepare_data.py --balanced_real_images 
-a /home/axe/Datasets/VQA_Dataset/v2_mscoco_train2014_annotations.json 
-q /home/axe/Datasets/VQA_Dataset/v2_OpenEnded_mscoco_train2014_questions.json 
-o /home/axe/Datasets/VQA_Dataset
```

This command creates the <i> vqa_dataset.txt </i> file in the `-o` output directory.

---
## Architecture

The architecture can be summarized as:-

Image --> CNN_encoder --> image_embedding <br>
Question --> LSTM_encoder --> question_embedding <br>

(image_embedding * question_embedding) --> FC --> Softmax --> answer_probability

![Alt text](vqa_baseline_architecture.png?raw=true "Baseline Architecture")

<br>
PyTorch Representation:

```
ContextNet(
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
