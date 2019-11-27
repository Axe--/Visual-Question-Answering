# VQA Baseline
Baseline VQA PyTorch implementation for Open-Ended Question-Answering



---
## Table of Contents

> The project comprises of the following sections.
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training](#training)
- [Experiment Logging](#experiment-logging)
- [Inference](#inference)
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
$ python3 prepare_data.py --balanced_real_images -s train \
-a /home/axe/Datasets/VQA_Dataset/raw/v2_mscoco_train2014_annotations.json \
-q /home/axe/Datasets/VQA_Dataset/raw/v2_OpenEnded_mscoco_train2014_questions.json \
-o /home/axe/Datasets/VQA_Dataset/processed/vqa_train2014.txt \
-v /home/axe/Datasets/VQA_Dataset/processed/vocab_count_5_K_1000.pickle -c 5 -K 1000  # vocab flags (for training set)
```

Stores the dataset file in the output directory `-o` and the corresponding vocab file `-v`. <br>
For validation/test sets, remove the vocabulary flags: `-v`, `-c`, `-K`.


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
      ...                                                   
      # Max-pool (5x)
      ...
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
$ python3 main.py --mode train --expt_name K_1000  --expt_dir /home/axe/Projects/VQA_baseline/results_log \
--train_img /home/axe/Datasets/VQA_Dataset/train2014 --train_file /home/axe/Datasets/VQA_Dataset/vqa_train2014.txt \
--val_img /home/axe/Datasets/VQA_Dataset/val2014 --val_file /home/axe/Datasets/VQA_Dataset/vqa_val2014.txt \
--gpu_id 0 --num_epochs 50 --batch_size 256 --num_cls 1000 --save_interval 1000 --log_interval 100 \
--run_name demo_run -lr 1e-4 --opt_lvl 1 --num_workers 2

```
Specify `--model_ckpt` (filename.pth) to load model checkpoint from disk <i>(resume training/inference)</i>

> *Note*: Setting num_cls (K) = 2 is equivalent to 'yes/no' setup. <br>
          For K > 2, it is an open-ended set.

### Experiment Logging

The experiment output log directory is structured as follows:

```
├── main.py
..
..
├── expt_dir
│   └── expt_name
│       ├── vocab_K.pkl
│       └── run_name
│           ├── events.out.tfevents.1572463346.axe-H270-Gaming-3
│           ├── model_4000.pth
│           └── train_log.txt

```


- **Option 1**
    - 🍴 ..

- **Option 2**
    - 👯 ..

### Inference 

- **....**


---


> *TODO*: Test with BERT embeddings (Pre-Trained & Fine-Tuned)


- [x] TensorBoardX
- [ ] BERT Embeddings

---

## References
[1]  [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468)
