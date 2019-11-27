"""
Given the VQA annotations & questions file, generates a dataset file (.txt) in the following format:

`image_name` <tab> `question` <tab> `answer`

* question & answer are comma-separated strings.

The resulting file is stored in the --output_dir
"""

"""
Training set:
$ python3 prepare_data.py --balanced_real_images -s train \
-a /home/axe/Datasets/VQA_Dataset/raw/v2_mscoco_train2014_annotations.json \
-q /home/axe/Datasets/VQA_Dataset/raw/v2_OpenEnded_mscoco_train2014_questions.json \
-o /home/axe/Datasets/VQA_Dataset/processed/vqa_train2014.txt \
-v /home/axe/Datasets/VQA_Dataset/processed/vocab_count_5_K_1000.pkl -c 5 -K 1000
"""

"""
Validation set:
python3 prepare_data.py --balanced_real_images -s val \
-a /home/axe/Datasets/VQA_Dataset/raw/v2_mscoco_val2014_annotations.json \
-q /home/axe/Datasets/VQA_Dataset/raw/v2_OpenEnded_mscoco_val2014_questions.json \
-o /home/axe/Datasets/VQA_Dataset/processed/vqa_val2014.txt
"""

import argparse
import os
from datahelper import VQA as DataHelper
from utils import save_vocab


def pad_with_zero(num, arg):
    total_digits = 6 if arg.balanced_real_images else 5
    num_zeros = total_digits - len(str(num))
    return num_zeros * "0" + str(num)


parser = argparse.ArgumentParser(description='Prepare data for balanced real images QA aka COCO')

# Dataset params
parser.add_argument('-s',  '--split',          type=str,   help='split set', required=True, choices=['train', 'val'])
parser.add_argument('-a',  '--annot_file',     type=str,   help='path to annotations file (.json)', required=True)
parser.add_argument('-q',  '--ques_file',      type=str,   help='path to questions file (.json)', required=True)
parser.add_argument('-o',  '--output_file',    type=str,   help='output (img, ques, ans) dataset file .txt', required=True)

# Vocab params (only to be used with training set)
parser.add_argument('-v',  '--vocab_file',     type=str,   help='output training set vocabulary file (.pkl)')
parser.add_argument('-c',  '--min_word_count', type=int,   help='min. word frequency for including in vocab', default=5)
parser.add_argument('-K',  '--num_cls',        type=int,   help='top-K most frequent answers as labels', default=1000)

group = parser.add_mutually_exclusive_group()
group.add_argument("--balanced_real_images", action="store_true",
                   help="image format is COCO_train2014_000000xxxxxx.jpg")

group.add_argument("--abstract_scene_images", action="store_true",
                   help="image format is abstract_v002_train2015_0000000xxxxx.png")

args = parser.parse_args()

image_prefix = ""
image_postfix = ""
assert (args.balanced_real_images != args.abstract_scene_images)
if args.balanced_real_images:
    if args.split == 'train':
        image_prefix = "COCO_train2014_000000"
    else:
        image_prefix = "COCO_val2014_000000"
    image_postfix = ".jpg"

elif args.abstract_scene_images:
    if args.split == 'train':
        image_prefix = "abstract_v002_train2015_0000000"
    else:
        raise NotImplementedError()

    image_postfix = ".png"

helper = DataHelper(args.annot_file, args.ques_file)

# Write dataset to file
with open(args.output_file, "w") as output_file:
    for i in range(len(helper.dataset['annotations'])):

        imd_id = helper.dataset['annotations'][i]['image_id']
        img_name = image_prefix + pad_with_zero(imd_id, args) + image_postfix

        ques_id = helper.dataset['annotations'][i]['question_id']
        question = helper.qqa[ques_id]['question']

        # Convert to comma-separated token string
        question = ','.join(question.strip().split())

        answer = helper.dataset['annotations'][i]['multiple_choice_answer']

        # each line contains: image_filename [tab] question [tab] answer
        output_file.write(img_name + "\t" + question + "\t" + answer + "\n")

print('Saved dataset file at: {}'.format(args.output_file))

# Read the newly created dataset file to build the vocabulary & save to disk
if args.vocab_file:
    save_vocab(args.output_file, args.vocab_file, args.min_word_count, args.num_cls)
