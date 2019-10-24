from torch.utils.data import Dataset
import torch
import skimage.io as io
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np
import os
from utils import build_vocab, preprocess_text, pad_sequences


class VQADataset(Dataset):
    """VQA Dataset"""

    def __init__(self, dataset_file, img_dir, K, transform=None):
        self.dataset_file_path = dataset_file
        self.images_dir = img_dir

        # Calculate the K most frequent answers from the dataset
        labels = self.frequent_answers(self.dataset_file_path, K)

        # Filter out samples which don't have answer in the top-K labels
        self.data = self.filter_samples_by_label(self.dataset_file_path, labels)

        # Map labels to indexes
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(labels)}

        # Build question vocabulary -- word2idx & idx2word mappings
        self.word2idx, self.idx2word, self.max_len_sequence = build_vocab(self.data)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Parse the text file line
        img_name, question, answer = self.data[idx].strip().split('\t')

        img_path = os.path.join(self.images_dir, img_name)

        image = resize(io.imread(img_path), [224, 224])

        question = preprocess_text(question)
        question = ['<START>'] + question + ['<END>']
        question = [self.word2idx[word] for word in question]
        question = pad_sequences(question, self.max_len_sequence)

        # Actual length of the sequence (ignoring padded elements)
        # used later by pad_packed_sequence (torch.nn.utils.rnn)
        ques_len = sum(1 - np.equal(question, 0))

        label_idx = self.label_to_idx[answer]

        # Collate for transforms
        img_ques_ans = {'image': image,
                        'question': question,
                        'ques_len': ques_len,
                        'label': label_idx}

        if self.transform:
            img_ques_ans = self.transform(img_ques_ans)

        return img_ques_ans

    @staticmethod
    def frequent_answers(file_path, K):
        """
        We treat answer tokens as class labels;
        the top K most frequent answers are selected

        :param file_path: path to dataset file
        :param K: num of labels
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

            top_k_answers = sorted(answer_frequency_dict.items(), key=lambda kv: kv[1])[:K]
            top_k_answers = [ans for ans, cnt in top_k_answers]

            return top_k_answers

    @staticmethod
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


# TODO:: Data Augmentation - Image Transforms
class ToTensor(object):
    """Convert numpy arrays to Tensors"""

    def __call__(self, sample):
        img = sample['image']

        # If image doesn't have color channels, add dummy rgb (axis)
        if len(img.shape) == 2:
            img = gray2rgb(img)

        # swap color axis because -> numpy image: H x W x C  # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        sample['image'] = torch.FloatTensor(img)

        return {'image': sample['image'],
                'question': sample['question'],
                'ques_len': sample['ques_len'],
                'label': sample['label']}
