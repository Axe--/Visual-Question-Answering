import torch
import torch.nn as nn
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import apex.amp as amp
from time import time
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tensorboardX import SummaryWriter
from model import VQABaselineNet
from utils import sort_batch, build_vocab
from dataloader import VQADataset
from dataloader import fetch_frequent_answers, filter_samples_by_label
from torch.optim.lr_scheduler import StepLR

"""
Train (with validation):
python3 main.py --mode train --expt_name sample_model_K_2_yes_no --expt_dir /home/axe/Projects/VQA_baseline/results_log 
--train_img /home/axe/Datasets/VQA_Dataset/train2014 --train_file /home/axe/Datasets/VQA_Dataset/vqa_dataset.txt 
--val_img /home/axe/Datasets/VQA_Dataset/train2014 --val_file /home/axe/Projects/VQA_baseline/sample_data.txt  
--gpu_id 1 --num_epochs 50 --batch_size 256 --num_cls 2 --save_interval 1000 --log_interval 100 --run_name demo_new_schdl 
--lr 1e-4
Test:
"""

PATH_VGG_WEIGHTS = '/home/axe/Projects/Pre_Trained_Models/vgg11_bn-6002323d.pth'


def main():
    parser = argparse.ArgumentParser(description='Visual Question Answering')

    # Experiment params
    parser.add_argument('--mode',       type=str,   help='train or test', default=True)
    parser.add_argument('--expt_dir',   type=str,   help='root directory to save model & summaries', required=True)
    parser.add_argument('--expt_name',  type=str,   help='expt_dir/expt_name: organize experiments', required=True)
    parser.add_argument('--run_name',   type=str,   help='expt_dir/expt_name/run_name: organize training runs', required=True)
    parser.add_argument('--val_size',   type=int,   help='validation set size for evaluating accuracy', default=10000)

    # Data params
    parser.add_argument('--train_img',  type=str,   help='path to training images directory', required=True)
    parser.add_argument('--train_file', type=str,   help='training dataset file', required=True)
    parser.add_argument('--val_file',   type=str,   help='validation dataset file')
    parser.add_argument('--val_img',    type=str,   help='path to validation images directory')

    # Vocab params
    parser.add_argument('--num_cls',       '-K',    type=int_min_two, help='top K answers (labels); min = 2', default=1000)
    parser.add_argument('--skip_yes_no',   '-skp',  type=str2bool,    help='exclude yes/no questions (class imbalance)', default='false')

    # Training params
    parser.add_argument('--batch_size',    '-bs',   type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',    '-ep',   type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate', '-lr',   type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--log_interval',           type=int,   help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval',          type=int,   help='save model after `n` weight update steps', default=3000)

    # Model params
    parser.add_argument('--model_ckpt',       type=str,       help='resume training/perform inference; e.g. model_1000.pth')
    parser.add_argument('--vgg_wts_path',     type=str,       help='VGG-11 (bn) pre-trained weights (.pth) file', default=PATH_VGG_WEIGHTS)
    parser.add_argument('--is_vgg_trainable', type=str2bool,  help='whether to train the VGG encoder', default='false')
    # parser.add_argument('--model_config', type=str, help='model config file - specifies model architecture')

    # GPU params
    parser.add_argument('--num_gpus',   type=int,   help='number of GPUs to use for training', default=1)
    parser.add_argument('--gpu_id',     type=int,   help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=0)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print('Selected Device: {}'.format(device))
    # torch.cuda.get_device_properties(device).total_memory  # in Bytes

    # Train params
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    # TODO: Multi-GPU PyTorch Implementation
    # if args.num_gpus > 1 and torch.cuda.device_count() > 1:
    #     print("Using {} GPUs!".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)

    # Train
    if args.mode == 'train':
        # Calculate the K most frequent answers from the dataset
        labels = fetch_frequent_answers(args.train_file, args.num_cls)

        # Due to class imbalance caused by 'yes' & 'no' classes, we can ignore them (questions) for open-ended answering
        if args.skip_yes_no and args.num_cls > 2:
            labels = [label for label in labels if label != 'yes' or label != 'no']

        print('Few top answers: {}\n'.format(', '.join(labels[:10])))

        # Filter out samples (I,Q,A) where the answer (A) is not in top-K labels
        train_data = filter_samples_by_label(args.train_file, labels)

        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        st = time()
        # Build/Load vocab  -->  /expt_dir/expt_name/
        vocab = preprocess_vocab(args, train_data, labels)
        print('\n ** Preprocessing time: {:.4f} secs.'.format(time() - st))

        # Unpack vocab
        word2idx, idx2word, label2idx, idx2label, max_seq_length = [v for k, v in vocab.items()]

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logs_file(parser, log_dir)

        # Dataset & Dataloader
        train_dataset = VQADataset(train_data, args.train_img, word2idx, label2idx, max_seq_length,
                                   transform=Compose([Resize((224, 224)), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                                                (0.229, 0.224, 0.225))]))
        # TODO: Data Augmentation - Image Transformations

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

        print('Question Vocabulary Size: {} \n\n'.format(len(word2idx.keys())))

        print('Train Data Size: {}'.format(train_dataset.__len__()))

        # Plot data (image, question, answer) for sanity check
        # plot_data(train_loader, idx2word, idx2label, num_plots=80)
        # sys.exit()

        if args.val_file:
            # Filter samples from the validation set, using top K labels from the training set
            val_data = filter_samples_by_label(args.val_file, labels)

            # Use the same word-index dicts as that obtained for the training set
            val_dataset = VQADataset(val_data, args.val_img, word2idx, label2idx, max_seq_length,
                                     transform=Compose([Resize((224, 224)), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                                                  (0.229, 0.224, 0.225))]))

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)
            print('Validation Data Size: {}'.format(val_dataset.__len__()))
            print('Accuracy on the validation set is computed using {} samples. See --val_size\n'.format(args.val_size))

        # Question Encoder params
        vocabulary_size = len(train_dataset.word2idx.keys())
        word_embedding_dim = 300
        encoder_hidden_units = 1024

        question_encoder_params = {'vocab_size': vocabulary_size, 'inp_emb_dim': word_embedding_dim,
                                   'enc_units': encoder_hidden_units, 'batch_size': batch_size}

        # Image Encoder params
        is_vgg_trainable = args.is_vgg_trainable  # default = False
        vgg_wts_path = args.vgg_wts_path  # default = PATH_VGG_WTS

        image_encoder_params = {'is_trainable': is_vgg_trainable, 'weights_path': vgg_wts_path}

        # Define model & load to device
        model = VQABaselineNet(question_encoder_params, image_encoder_params, K=args.num_cls)
        model.to(device)

        # Load model checkpoint file (if specified) from `log_dir`
        if args.model_ckpt:
            model_ckpt_path = os.path.join(log_dir, args.model_ckpt)
            checkpoint = torch.load(model_ckpt_path)

            model.load_state_dict(checkpoint)

            log_msg = 'Model successfully loaded from {}'.format(model_ckpt_path) + '\nResuming Training...'

            log_file.write(log_msg)
            log_file.flush()
            print(log_msg)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)

        # TODO: StepLR Scheduler
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        # TODO: Mixed-Precision training (amp support)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        steps_per_epoch = len(train_loader)
        start_time = time()
        curr_step = 0
        # TODO: Save model with best validation accuracy
        best_val_acc = 0.0

        for epoch in range(n_epochs):
            for batch_data in train_loader:
                # Load batch data
                image = batch_data['image']
                question = batch_data['question']
                ques_len = batch_data['ques_len']
                label = batch_data['label']

                # Sort batch based on sequence length
                image, question, label, ques_len = sort_batch(image, question, label, ques_len)

                # Set `question` to sequence-first --> swap: (batch x seq) -> (seq x batch)
                question = question.transpose(1, 0)

                # Load data onto the available device
                image = image.to(device)
                question = question.to(device)
                ques_len = ques_len.to(device)
                label = label.to(device)

                # Forward Pass
                label_predict = model(image, question, ques_len, device)

                # Compute Loss
                loss = criterion(label_predict, label)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                # TODO: loss.backward() becomes:
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if (curr_step + 1) % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if args.val_file:
                        validation_accuracy = compute_accuracy(model, val_loader, device, size=args.val_size)

                        log_msg = 'Validation Accuracy: {:.2f} %'.format(validation_accuracy)

                        log_file.write(log_msg + '\n')
                        log_file.flush()

                        print(log_msg)

                        # If current model has the best accuracy on the validation set & >= training accuracy,
                        # save model to disk

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Accuracy', validation_accuracy, curr_step)

                        # Reset the mode to training
                        model.train()

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    # total time = time_per_step * steps_per_epoch * total_epochs
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                        epoch + 1, n_epochs, curr_step + 1, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    log_file.write(log_msg + '\n')
                    log_file.flush()

                    print(log_msg)

                # Save the model
                if (curr_step + 1) % args.save_interval == 0:
                    print('Saving the model at the {} step to directory:{}'.format(curr_step + 1, log_dir))
                    save_path = os.path.join(log_dir, 'model_' + str(curr_step + 1) + '.pth')
                    torch.save(model.state_dict(), save_path)

                curr_step += 1

        writer.close()
        log_file.close()

    # Test
    """
    elif args.mode == 'test':
        test_dataset = VQADataset(????)

        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)

        checkpoint = torch.load(args.model_ckpt_file)

        # TODO: Retrieve Params from trained model (checkpoint file)
        # Question Encoder params
        vocabulary_size = -1
        word_embedding_dim = 256
        encoder_hidden_units = 1024

        question_encoder_params = {'vocab_size': vocabulary_size, 'inp_emb_dim': word_embedding_dim,
                                   'enc_units': encoder_hidden_units, 'batch_size': batch_size}

        # Image Encoder params
        is_vgg_trainable = args.is_vgg_trainable        # default = False
        vgg_wts_path = args.vgg_wts_path                # default = PATH_VGG_WTS

        image_encoder_params = {'is_trainable': is_vgg_trainable, 'weights_path': vgg_wts_path}

        # Define model & load to device
        model = VQABaselineNet(question_encoder_params, image_encoder_params)
        model.to(device)

        # Load pre-trained weights for validation
        model.load_state_dict(checkpoint)
        print('Model successfully loaded from {}'.format(args.model_ckpt_file))

        # Compute test accuracy
        test_accuracy = compute_accuracy(model, test_loader, device, show_preds=True, mode='Test')
        print('Test Accuracy: {:.2f} %'.format(test_accuracy))
    """


def compute_accuracy(model, dataloader, device, size):
    """
    For the given model, computes accuracy on validation/test set

    :param model: VQA model
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to compute accuracy
    :return: None
    """
    model.eval()
    with torch.no_grad():
        num_correct = 0
        total = 0

        n_iters = size // dataloader.batch_size

        # Evaluate on mini-batches
        for i, batch in enumerate(dataloader):
            # Load batch data
            image = batch['image']
            question = batch['question']
            ques_len = batch['ques_len']
            label = batch['label']

            # Sort batch based on sequence length
            image, question, label, ques_len = sort_batch(image, question, label, ques_len)

            # Set `question` to sequence-first --> swap: (batch x seq) -> (seq x batch)
            question = question.transpose(1, 0)

            # Load data onto the available device
            image = image.to(device)
            question = question.to(device)
            ques_len = ques_len.to(device)
            label = label.to(device)

            # Forward Pass
            label_logits = model(image, question, ques_len, device)

            # Compute Accuracy
            label_predicted = torch.argmax(label_logits, dim=1)
            correct = (label == label_predicted)

            num_correct += correct.sum().item()
            total += len(label)

            if i >= n_iters:
                break

        accuracy = 100.0 * num_correct / total

        return accuracy


def preprocess_vocab(args, train_data, labels):
    """
    Given training dataset and data filter flags (K, skip_yes_no), builds vocabulary from training set. \n
    Saves vocab to /args.expt_dir/args.expt_name/file_name.pkl. \n

    If file exists for the current data configuration (args.num_cls, args.skip_yes_no),
    reads vocab from disk.

    :param args: argparse arguments
    :param train_data: filtered list of image_filename question answer triplets
    :param labels: filtered list of answers (class labels)

    :return: dict {word2idx, idx2word, label2idx, idx2label, max_seq_length}
    """
    # Save/Read word-idx dicts & label-idx dicts to disk: /expt_dir/expt_name/
    expt_name_dir = os.path.join(args.expt_dir, args.expt_name)

    skip_yes_no = '_skip_yes_no' if args.skip_yes_no and args.num_cls > 2 else ''

    # Stores: word_idx_dicts, label_idx_dicts
    vocab_filename = 'vocab_K_{}{}.pkl'.format(args.num_cls, skip_yes_no)
    vocab_file_path = os.path.join(expt_name_dir, vocab_filename)

    # If vocab previously created, load from disk
    if os.path.exists(vocab_file_path):
        with open(vocab_file_path, 'rb') as handle:
            vocab = pickle.load(handle)

            print('Loading vocab data from {}'.format(vocab_file_path))

        return vocab

    # Else build vocab from training data
    else:
        # Build question vocab (word-index dicts)
        word2idx, idx2word, max_seq_length = build_vocab(train_data)

        # Map answer labels to indexes (label-index dicts)   {idx <---> 'answer'}
        label2idx = {label: idx for idx, label in enumerate(labels)}
        idx2label = {idx: label for idx, label in enumerate(labels)}

        # word-index & label-index dicts
        vocab = {'word2idx': word2idx, 'idx2word': idx2word,
                 'label2idx': label2idx, 'idx2label': idx2label,
                 'max_seq_length': max_seq_length}

        # Save vocab to disk
        with open(vocab_file_path, 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print('Saving vocab data at {}'.format(vocab_file_path))

        return vocab


def setup_logs_file(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + os.path.basename(__file__) + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def plot_data(dataloader, idx2word, idx2label, num_plots=4):
    """
    For plotting input data (after preprocessing with dataloader). \n
    Helper for sanity check.
    """
    for i, data in enumerate(dataloader):
        st = time()

        # Read dataset, select one random sample from the mini-batch
        batch_size = len(data['label'])
        idx = np.random.choice(batch_size)
        ques = data['question'][idx]
        label = data['label'][idx]
        img = data['image'][idx]
        print('Fetch Time {:.4f} secs'.format(time() - st))

        # Convert question tokens to words & answer class index to label
        ques_str = ' '.join([idx2word[word_idx] for word_idx in ques.tolist()])
        ans_str = ' '.join(idx2label[label.tolist()])

        st = time()
        # Plot Data
        plt.imshow(img.permute(1, 2, 0))
        plt.text(0, 0, ques_str, bbox=dict(fill=True, facecolor='white', edgecolor='red', linewidth=2))
        plt.text(220, 220, ans_str, bbox=dict(fill=True, facecolor='white', edgecolor='blue', linewidth=2))
        plt.show()
        print('Plot Time {:.4f} secs'.format(time() - st))

        i += 1

        if i >= num_plots:
            break


def str2bool(v):
    v = v.lower()
    assert v == 'true' or v == 'false'
    return v.lower() == 'true'


def int_min_two(k):
    k = int(k)
    assert k >= 2 and type(k) == int, 'Ensure k >= 2'
    return k


if __name__ == '__main__':
    main()