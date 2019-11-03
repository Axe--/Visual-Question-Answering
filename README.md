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

> Image_Encoder() --> img_emb       <br>
  Question_Encoder() --> ques_emb   <br>
  img_emb * ques_emb --> MLP() --> pred_cls

```
VQABaselineNet(
  (image_encoder): ImageEncoder(
    (vgg11_encoder): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU(inplace=True)
      (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (10): ReLU(inplace=True)
      (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (13): ReLU(inplace=True)
      (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (17): ReLU(inplace=True)
      (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (20): ReLU(inplace=True)
      (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (24): ReLU(inplace=True)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (27): ReLU(inplace=True)
      (28): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (29): AdaptiveAvgPool2d(output_size=(7, 7))
      (30): Flatten()
      (31): Linear(in_features=25088, out_features=4096, bias=True)
      (32): ReLU(inplace=True)
      (33): Dropout(p=0.5, inplace=False)
      (34): Linear(in_features=4096, out_features=4096, bias=True)
    )
    (embedding_layer): Sequential(
      (0): Linear(in_features=4096, out_features=1024, bias=True)
      (1): Tanh()
    )
  )
  (question_encoder): QuestionEncoder(
    (word_embedding_matrix): Sequential(
      (0): Embedding(10, 300)
      (1): Tanh()
    )
    (gru): GRU(300, 1024)
    (embedding_layer): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (mlp): Sequential(
    (0): Linear(in_features=1024, out_features=1000, bias=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): Tanh()
    (3): Linear(in_features=1000, out_features=2, bias=True)
    (4): Dropout(p=0.5, inplace=False)
  )
  (cls_predict): Softmax(dim=1)
)

```

---

## Training

Run the following script for training:

```bash
$ python3 main.py --mode train --model_name sample_model --train_img /home/axe/Datasets/VQA_Dataset/train2014 \
--train_file /home/axe/Datasets/VQA_Dataset/vqa_dataset.txt --val_file /home/axe/Projects/VQA_baseline/sample_data.txt \
--val_img /home/axe/Datasets/VQA_Dataset/train2014 --log_dir /home/axe/Projects/VQA_baseline/results_log --gpu_id 1 \
--num_epochs 50 --batch_size 256 --num_cls 1000 --save_after 500 --log_interval 100 --expt_name demo_1000 --learning_rate 1e-4
```

> *Note* : Setting num_cls = 2 is equivalent to 'yes/no' setup. (K = 2)
For K > 2, it is an open-ended set.

---


> *TODO* : Add TensorBoardX support


- [ ] TensorBoardX
- [ ] BERT Embeddings (huggingface)

---

## References
[1]  [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468)
