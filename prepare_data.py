import argparse
import os
from datahelper import VQA as DataHelper


def pad_with_zero(num):
    total_digits = 6 if args.balanced_real_images else 5
    num_zeros = total_digits - len(str(num))
    return num_zeros * "0" + str(num)


parser = argparse.ArgumentParser(description='Prepare data for balanced real images QA aka COCO')

parser.add_argument('-a', '--annot_file', type=str, help='path to annotations file (.json)',
                    required=True)
parser.add_argument('-q', '--ques_file', type=str, help='path to questions file (.json)',
                    required=True)
parser.add_argument('-o', '--output_dir', type=str, help='stores the dataset file (img, ques, ans) - vqa_dataset.txt',
                    required=True)

group = parser.add_mutually_exclusive_group()
group.add_argument("--balanced_real_images", action="store_true",
                   help="image format is COCO_train2014_000000xxxxxx.jpg")

group.add_argument("--abstract_scene_images", action="store_true",
                   help="image format is abstract_v002_train2015_0000000xxxxx.png")

args = parser.parse_args()

# image_prefix = args.input_dir + "images/"
image_prefix = ""
image_postfix = ""
assert (args.balanced_real_images != args.abstract_scene_images)
if args.balanced_real_images:
    image_prefix += "COCO_train2014_000000"
    image_postfix = ".jpg"
elif args.abstract_scene_images:
    image_prefix += "abstract_v002_train2015_0000000"
    image_postfix = ".png"

helper = DataHelper(args.annot_file, args.ques_file)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_file_path = os.path.join(args.output_dir, "vqa_dataset.txt")

# each line contains: image_filename[tab]question[tab]answer
with open(output_file_path, "w") as output_file:
    for i in range(len(helper.dataset['annotations'])):

        imd_id = helper.dataset['annotations'][i]['image_id']
        img_name = image_prefix + pad_with_zero(imd_id) + image_postfix

        ques_id = helper.dataset['annotations'][i]['question_id']
        question = helper.qqa[ques_id]['question']

        # Convert to comma-separated token string
        question = ','.join(question.strip().split())

        answer = helper.dataset['annotations'][i]['multiple_choice_answer']

        output_file.write(img_name + "\t" + question + "\t" + answer + "\n")
