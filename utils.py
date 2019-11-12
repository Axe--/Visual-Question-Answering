"""
Util functions for:
- Building vocabulary (word2idx & idx2word)
- Pre-processing text data (punctuation, tokenize, etc.)
"""
import string
import numpy as np


def pad_sequences(seq, max_len):
    """
    Pads a sequence, given max length
    :param seq: list (int tokens)
    :param max_len: pad to max length
    :return: list (padded sequence)
    """

    padded = np.zeros((max_len,), np.int64)
    if len(seq) > max_len:
        padded[:] = seq[:max_len]
    else:
        padded[:len(seq)] = seq
    return padded


def sort_batch(images, questions, answers, ques_seq_lens):
    """
    Sort data (desc.) based on sequence lengths of batch sample (needed for pad_packed_sequence)
    :param images:
    :param questions:
    :param answers:
    :param ques_seq_lens:
    :return:
    """
    # question --> (batch_size, sequence_length)

    ques_seq_lens, idx = ques_seq_lens.sort(dim=0, descending=True)
    questions = questions[idx]
    answers = answers[idx]
    images = images[idx]

    return images, questions, answers, ques_seq_lens


def preprocess_text(text):
    """
    Given comma-separated text, removes punctuations & converts to lowercase.

    :param text: string of comma-separated words (sentence)
    :return: array of tokens
    """
    # Comma-separated word tokens
    text_token_list = text.strip().split(',')
    text = ' '.join(text_token_list)

    # Remove punctuations
    table = str.maketrans('', '', string.punctuation)
    words = text.strip().split()
    words = [w.translate(table) for w in words]

    # Set to lowercase & drop empty strings
    words = [word.lower() for word in words if word != '' and word != 's']

    return words


def build_vocab(data):
    """
    Given the VQA Dataset, builds vocabulary for the questions

    :param data: img_name \t question \t answer
    :return: index2word, word2index & max sequence length
    """
    vocab = set()
    max_sequence_length = 0

    # Build a set of unique words
    for sample in data:
        question = sample.split('\t')[1].strip()

        # Clean the text input & convert from string to list
        words = preprocess_text(question)

        # Add words to vocabulary
        vocab = vocab.union(set(words))

        # Update the max length sequence in the dataset
        if len(words) > max_sequence_length:
            max_sequence_length = len(words)

    # Build word to index mapping
    helper_tokens = {'<PAD>': 0, '<UNKNOWN>': 1}

    vocab = {word: idx + 2 for idx, word in enumerate(list(vocab))}

    word2idx = {**helper_tokens, **vocab}

    # Conversely index to word mapping
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, max_sequence_length
