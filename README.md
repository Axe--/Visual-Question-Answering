# Visual Question Answering
VQA implementations in PyTorch for Open-Ended Question-Answering
- [Baseline <i>(LSTM Q + I)</i>](#references)
- [Hierarchical Co-Attention <i>(Parallel)</i>](#references)


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


### Baseline


The architecture can be summarized as:-

Image --> CNN_encoder --> <i>image_embedding</i> <br>
Question --> LSTM_encoder --> <i>question_embedding</i> <br>

(image_embedding * question_embedding) --> MLP_Classifier --> <i>answer_logit</i>

![Baseline](assets/vqa_baseline_architecture.png?raw=true "Baseline Architecture")

<br>



### Hierarchical Co-Attention

The architecture can be summarized as:-

Image --> CNN_encoder --> <i>image_embedding</i> <br>
Question --> Word_Emb --> Phrase_Conv_MaxPool --> Sentence_LSTM --> <i>question_embedding</i> <br>

ParallelCoAttention( image_embedding, question_embedding ) --> MLP_Classifier --> <i>answer_logit</i> 

![HieCoAttn](assets/hiecoattn.png?raw=true "HieCoAttn Architecture")

![Parallel](assets/parallel_attn.png?raw=true "HieCoAttn Architecture")


---

## Training

Run the following script for training:

```bash
$ python3 main.py --mode train --expt_name K_1000_Attn --expt_dir /home/axe/Projects/VQA_baseline/results_log \
--train_img /home/axe/Datasets/VQA_Dataset/raw/train2014 --train_file /home/axe/Datasets/VQA_Dataset/processed/vqa_train2014.txt \
--val_img /home/axe/Datasets/VQA_Dataset/raw/val2014 --val_file /home/axe/Datasets/VQA_Dataset/processed/vqa_val2014.txt\
--vocab_file /home/axe/Datasets/VQA_Dataset/processed/vocab_count_5_K_1000.pickle --save_interval 1000 \
--log_interval 100 --gpu_id 0 --num_epochs 50 --batch_size 160 -K 1000 -lr 1e-4 --opt_lvl 1 --num_workers 6 \
--run_name O1_wrk_6_bs_160 --model attention

```
Specify `--model_ckpt` (filename.pth) to load model checkpoint from disk <i>(resume training/inference)</i> <br>

Select the architecture by using `--model` ('baseline', 'attention'). <br>

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
│       └── run_name
│           ├── events.out.tfevents
│           ├── model_4000.pth
│           └── train_log.txt

```



### Inference 

- **....**


---


> *TODO*: Test with BERT embeddings (Pre-Trained & Fine-Tuned)


- [x] Baseline & HieCoAttn
- [ ] VQA w/ BERT
- [ ] Inference & Attention Visualization

---

## References
[1]  [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468) <br>
[2]  [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/pdf/1606.00061)
