from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from utils import preprocess_text, pad_sequences


class VQADataset(Dataset):
    """VQA Dataset"""

    def __init__(self, data_file, img_dir, word2idx, label2idx, max_seq_length, transform):
        """
        The params - `word2idx`, `label2idx`, `max_seq_length` are
        common across train, validation & test sets.

        Dataset file contains samples in the following format:
        `img_name \t question \t answer`

        :param data_file: dataset file path
        :param str img_dir: path to images directory
        :param dict word2idx: word to index mapping
        :param dict label2idx: answer labels to class index mapping  (for top K)
        :param int max_seq_length: length of the longest question (word sequence)
        :param transform: image transform functions (preprocessing + data augmentation)
        """
        self.data_file = data_file
        self.images_dir = img_dir
        self.label2idx = label2idx
        self.word2idx = word2idx
        self.max_sequence_length = max_seq_length

        # Image transforms
        self.transform = transform

        # Read dataset file
        with open(self.data_file, 'r') as f:
            self.data = f.read().strip().split('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Parse the text file line
        img_name, question, answer = self.data[idx].strip().split('\t')

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Resize(224, 224); ToTensor(); Normalize(mean, std_dev)
        image = self.transform(image)    # uint8 --> float32

        # Preprocess question (str --> list)
        question = preprocess_text(question)

        # Convert caption words to indexes
        # Map words not in the training set's vocab to <UNKNOWN>
        question = [self.word2idx[word] if word in self.word2idx else self.word2idx['<UNKNOWN>'] for word in question]

        # Pad question to max sequence length
        question = pad_sequences(question, self.max_sequence_length)

        # Actual length of the sequence (ignoring padded elements)
        # used later by pad_packed_sequence (torch.nn.utils.rnn)
        ques_len = sum(1 - np.equal(question, 0))

        # Convert answer to label index
        # If current answer is not in the answer vocab, replace it with the `UNKNOWN` label
        label_idx = self.label2idx[answer if answer in self.label2idx else 'UNKNOWN']

        # Collate for transforms
        img_ques_ans = {'image': image, 'question': question, 'ques_len': ques_len, 'label': label_idx}

        return img_ques_ans
