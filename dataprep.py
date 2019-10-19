import argparse
from datahelper import VQA as Datahelper
import pickle

parser = argparse.ArgumentParser(description='Prepare data for balanced real images QA aka COCO')

parser.add_argument('-i', '--inputDir', type=str, help='path to ../balanced real/ directory, must include / at the end', required=True)
parser.add_argument('-L', '--labelDir', type=str, help='directory to store first 1000 most frequent answers', required=True)
parser.add_argument('-n', '--numLabel', type=str, help='number of labels to store in labelDir/labels.pickle', required=True)
parser.add_argument('-d', '--dictDir', type=str, help='directory to store index2word and word2index dictionaries', required=True)
parser.add_argument('-o', '--outputDir', type=str, help='output directory, must include / at the end', required=True)

group = parser.add_mutually_exclusive_group()
group.add_argument("--balanced_real_images", action = "store_true", help = "image format is COCO_train2014_000000xxxxxx.jpg")
group.add_argument("--abstract_scene_images", action = "store_true", help = "image format is abstract_v002_train2015_0000000xxxxx.png")

args = parser.parse_args()

annFile = args.inputDir + "annotations.json"
quesFile = args.inputDir + "questions.json"

# image_prefix = args.inputDir + "images/"
image_prefix = ""
image_postfix = ""
assert(args.balanced_real_images != args.abstract_scene_images)
if args.balanced_real_images:
    image_prefix += "COCO_train2014_000000"
    image_postfix = ".jpg"
elif args.abstract_scene_images:
    image_prefix += "abstract_v002_train2015_0000000"
    image_postfix = ".png"


def padWithZero(num):
    totalDigits = 6 if args.balanced_real_images else 5
    numZeros = totalDigits - len(str(num))
    return numZeros * "0" + str(num)

helper = Datahelper(annFile, quesFile)

imgQuesAnsTupList = []

word2Index = {
    '<START>': 0,
    '<END>': 1,
    '<PAD>': 2,
}
ansWordCount = {
    0: 0,
    1: 0,
    2: 0,
}
num_unique_words = 3

fileLoc = args.outputDir + "vqa.txt"
mainOutputFile = open(fileLoc, "a")

for i in range(len(helper.dataset['annotations'])):

    imgID = helper.dataset['annotations'][i]['image_id']
    imgStr = image_prefix + padWithZero(imgID) + image_postfix


    quesID = helper.dataset['annotations'][i]['question_id']
    quesStr = helper.qqa[quesID]['question']

    ansStr = helper.dataset['annotations'][i]['multiple_choice_answer']

    mainOutputFile.write(imgStr + "\t" + quesStr + "\t" + ansStr + "\n")


    for word in quesStr.split():
        if word not in word2Index:
            word2Index[word] = num_unique_words
            num_unique_words += 1

    for word in ansStr.split():
        if word not in word2Index:
            word2Index[word] = num_unique_words
            num_unique_words += 1
        if word in ansWordCount:
            ansWordCount[word] += 1
        else:
            ansWordCount[word] = 0

# each line contains: image_filename[tab]question[tab]answer
mainOutputFile.close()

# save label
sortedLabels = sorted(ansWordCount, key=lambda k: ansWordCount[k])[:args.numLabel]
with open(args.labelDir + 'labelList', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(sortedLabels, filehandle)

# save index2word and word2index
index2Word = {v: k for k, v in word2Index.items()}
with open(args.dictDir + 'index2Word_dict', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(index2Word, filehandle)
with open(args.dictDir + 'word2Index_dict', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(word2Index, filehandle)

# index to one-hot: use pytorch to generate