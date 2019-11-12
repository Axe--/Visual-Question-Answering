from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from time import time
from utils import preprocess_text, pad_sequences


class VQADataset(Dataset):
    """VQA Dataset"""

    def __init__(self, data, img_dir, label2idx, max_seq_length, word2idx, transform):
        """
        :param data: list of filtered dataset samples ("img_name \t question \t answer")
        :param label2idx: answer labels to class index mapping  (for top K)
        :param img_dir: path to images directory
        :param max_seq_length: length of the longest question (word sequence)
        :param word2idx: word to index mapping (common across train, validation & test sets)
        :param transform: image transform functions (preprocessing + data augmentation)
        """
        self.data = data
        self.images_dir = img_dir
        self.label2idx = label2idx
        self.word2idx = word2idx
        self.max_sequence_length = max_seq_length

        # Image transforms
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # st = time()

        # Parse the text file line
        img_name, question, answer = self.data[idx].strip().split('\t')

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Resize (224, 224); ToTensor(); Normalize(mean, std_dev)
        image = self.transform(image)    # uint8 --> float32

        # print('Image Time {:.4f} secs'.format(time() - st))

        question = preprocess_text(question)
        question = [self.word2idx[word] for word in question]
        question = pad_sequences(question, self.max_sequence_length)

        # Actual length of the sequence (ignoring padded elements)
        # used later by pad_packed_sequence (torch.nn.utils.rnn)
        ques_len = sum(1 - np.equal(question, 0))

        label_idx = self.label2idx[answer]

        # Collate for transforms
        img_ques_ans = {'image': image,
                        'question': question,
                        'ques_len': ques_len,
                        'label': label_idx}

        # print('__getitem__() Time {:.4f} secs'.format(time() - st))

        return img_ques_ans


def fetch_frequent_answers(file_path, K):
    """
    We treat answer tokens as class labels;
    the top K most frequent answers are selected

    :param str file_path: path to dataset file
    :param int K: num of labels
    :return: `K` most frequent answers
    :rtype: list
    """
    with open(file_path, 'r') as file_in:
        answer_frequency_dict = {}

        line = file_in.readline()

        while line:
            answer = line.strip().split('\t')[2]

            if answer in answer_frequency_dict:
                answer_frequency_dict[answer] += 1
            else:
                answer_frequency_dict[answer] = 1

            line = file_in.readline()

        top_k_answers = sorted(answer_frequency_dict.items(), reverse=True, key=lambda kv: kv[1])[:K]
        top_k_answers = [ans for ans, cnt in top_k_answers]

        return top_k_answers


def filter_samples_by_label(file_path, labels):
    """
    Filters out samples that don't contain answers in the labels list

    :param file_path: path to dataset file
    :param labels: answer labels
    :type labels: list

    :return: filtered list of samples from the data file
    """
    # Convert to HashSet: O(1) lookup
    labels = set(labels)

    with open(file_path, 'r') as file_in:
        data = []

        line = file_in.readline()

        while line:
            answer = line.strip().split('\t')[2]

            if answer in labels:
                data.append(line)

            line = file_in.readline()

        return data
