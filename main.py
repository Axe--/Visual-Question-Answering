import torch
import torch.nn as nn
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tensorboardX import SummaryWriter
from model import VQABaselineNet
from utils import sort_batch, build_vocab
from dataloader import VQADataset
from dataloader import fetch_frequent_answers, filter_samples_by_label

"""
Train (with validation):
python3 main.py --mode train --model_name sample_model --train_img /home/axe/Datasets/VQA_Dataset/train2014 
--train_file /home/axe/Datasets/VQA_Dataset/vqa_dataset.txt --val_file /home/axe/Projects/VQA_baseline/sample_data.txt 
--val_img /home/axe/Datasets/VQA_Dataset/train2014 --log_dir /home/axe/Projects/VQA_baseline/results_log 
--gpu_id 1 --num_epochs 50 --batch_size 16 --num_cls 1000

Test:
"""

PATH_VGG_WEIGHTS = '/home/axe/Projects/Pre_Trained_Models/vgg11_bn-6002323d.pth'


def str2bool(v):
    v = v.lower()
    assert v == 'true' or v == 'false'
    return v.lower() == 'true'


def compute_accuracy(model, dataloader, device, mode='Validation'):
    """
    For the given model, computes accuracy on validation/test set

    :param model: VQA model
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param mode: validation or test
    :return: None
    """
    model.eval()
    with torch.no_grad():
        num_correct = 0
        total = 0

        # Evaluate on mini-batches
        for batch in dataloader:
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

        return 100.0 * num_correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Context Model')

    parser.add_argument('--mode',             type=str,      help='train or test', default='train')
    parser.add_argument('--model_name',       type=str,      help='model save ckpt folder', default='baseline_model')
    parser.add_argument('--train_img',        type=str,      help='path to training images directory', required=True)
    parser.add_argument('--train_file',       type=str,      help='train file', required=True)
    parser.add_argument('--val_file',         type=str,      help='validation file')
    parser.add_argument('--val_img',          type=str,      help='path to validation images directory')
    parser.add_argument('--num_cls',          type=int,      help='top K answers used as class labels', default=1000)
    parser.add_argument('--without_yes_no',   type=str2bool, help='Filters out yes/no questions', default='false')
    parser.add_argument('--batch_size',       type=int,      help='batch size', default=8)
    parser.add_argument('--num_epochs',       type=int,      help='number of epochs', default=50)
    parser.add_argument('--learning_rate',    type=float,    help='initial learning rate', default=1e-4)
    parser.add_argument('--num_gpus',         type=int,      help='number of GPUs to use for training', default=1)
    parser.add_argument('--gpu_id',           type=int,      help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=1)
    parser.add_argument('--log_dir',          type=str,      help='path to save model & summaries', required=True)
    parser.add_argument('--expt_name',        type=str,      help='summaries folder (within `--logdir`)', default='summaries')
    parser.add_argument('--log_interval',     type=int,      help='train step interval size for logging summaries', default=100)
    parser.add_argument('--save_after',       type=int,      help='save model after every `n` weight update steps', default=3000)
    parser.add_argument('--threshold_acc',    type=float,    help='threshold margin for validation/test accuracy', default=0.2)
    parser.add_argument('--margin_triplet',   type=float,    help='margin value for triplet loss', default=2.0)
    parser.add_argument('--model_ckpt_file',  type=str,      help='path to saved model checkpoint file (.pth)')
    parser.add_argument('--vgg_wts_path',     type=str,      help='VGG-11 (bn) pre-trained weights (.pth) file', default=PATH_VGG_WEIGHTS)
    parser.add_argument('--is_vgg_trainable', type=str2bool, help='whether to train the VGG encoder', default='false')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    print('Selected Device: {}'.format(device))

    # Hyper params
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    # TODO: Multi-GPU PyTorch Implementation
    # if args.num_gpus > 1 and torch.cuda.device_count() > 1:
    #     print("Using {} GPUs!".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)

    # Calculate the K most frequent answers from the dataset
    labels = fetch_frequent_answers(args.train_file, args.num_cls)

    print('Few top answers: {}'.format(', '.join(labels[:10])))

    # Due to class imbalance caused by 'yes' & 'no' classes, we can ignore them (questions) for open-ended answering
    if args.without_yes_no and args.num_cls > 2:
        labels = [label for label in labels if label != 'yes' or label != 'no']

    # Filter out samples which don't have answer in the top-K labels set
    train_data = filter_samples_by_label(args.train_file, labels)

    # Map labels to indexes
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for idx, label in enumerate(labels)}

    word2idx, idx2word, max_seq_length = build_vocab(train_data)

    word_idx_dicts = {'word2idx': word2idx, 'idx2word': idx2word}

    # Train
    if args.mode == 'train':
        # TensorBoard setup
        expt_dir = os.path.join(args.log_dir, args.model_name, args.expt_name)
        print('Training Log Directory: {}'.format(expt_dir))

        writer = SummaryWriter(expt_dir)

        # Dataset & Dataloader
        train_dataset = VQADataset(train_data, label_to_idx, args.train_img, max_seq_length, word_idx_dicts,
                                   transform=Compose([ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                            (0.229, 0.224, 0.225))]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

        print('Question Vocabulary Size: {} \n\n'.format(len(word2idx.keys())))

        print('Train Data Size: {}'.format(train_dataset.__len__()))

        """
        for sample_data in train_loader:
            # Read dataset
            ques = sample_data['question'][0]
            label = sample_data['label'][0]
            img = sample_data['image'][0]
        
            ques_str = ' '.join([train_dataset.idx2word[word] for word in ques.tolist()])
            ans_str = ' '.join(train_dataset.idx_to_label[label.tolist()])
        
            # Plot Data
            plt.imshow(img.permute(1, 2, 0))
            plt.text(0, 0, ques_str, bbox=dict(fill=True, facecolor='white', edgecolor='red', linewidth=2))
            plt.text(220, 220, ans_str, bbox=dict(fill=True, facecolor='white', edgecolor='blue', linewidth=2))
            plt.show()
        """

        if args.val_file:
            # Filter samples from the validation set, using top K labels from the training set
            val_data = filter_samples_by_label(args.val_file, labels)

            # Use the same word-index dicts as that obtained for the training set
            val_dataset = VQADataset(val_data, label_to_idx, args.val_img, max_seq_length, word_idx_dicts,
                                     transform=Compose([ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                              (0.229, 0.224, 0.225))]))

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)
            print('Validation Data Size: {}'.format(val_dataset.__len__()))

        # Question Encoder params
        vocabulary_size = len(train_dataset.word2idx.keys())
        word_embedding_dim = 300
        encoder_hidden_units = 1024

        question_encoder_params = {'vocab_size': vocabulary_size, 'inp_emb_dim': word_embedding_dim,
                                   'enc_units': encoder_hidden_units, 'batch_size': batch_size}

        # Image Encoder params
        is_vgg_trainable = args.is_vgg_trainable        # default = False
        vgg_wts_path = args.vgg_wts_path                # default = PATH_VGG_WTS

        image_encoder_params = {'is_trainable': is_vgg_trainable, 'weights_path': vgg_wts_path}

        # Define model & load to device
        model = VQABaselineNet(question_encoder_params, image_encoder_params, K=args.num_cls)
        model.to(device)

        # Load model checkpoint file (if specified)
        if args.model_ckpt_file:
            checkpoint = torch.load(args.model_ckpt_file)
            model.load_state_dict(checkpoint)
            print('Model successfully loaded from {}'.format(args.model_ckpt_file))

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        # Save path  -->  /log_dir/model_name/expt_name
        save_dir = os.path.join(args.log_dir, args.model_name, args.expt_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        steps_per_epoch = len(train_loader)
        start_time = time.time()
        curr_step = 0
        best_val_acc = 0.0   # TODO: Save model with best validation accuracy

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
                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if (curr_step + 1) % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if args.val_file:
                        validation_accuracy = compute_accuracy(model, val_loader, device)
                        print('Validation Accuracy: {:.2f} %'.format(validation_accuracy))

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Accuracy', validation_accuracy, curr_step)

                        # Reset the mode to training
                        model.train()

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time.time() - start_time) / 3600
                    # total time = time_per_step * steps_per_epoch * total_epochs
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                            epoch + 1, n_epochs, curr_step + 1, steps_per_epoch, loss.item(), time_elapsed, time_left))

                # Save the model
                if (curr_step + 1) % args.save_after == 0:
                    print('Saving the model at the {} step to directory:{}'.format(curr_step + 1, save_dir))
                    save_path = os.path.join(save_dir, 'model_' + str(curr_step + 1) + '.pth')
                    torch.save(model.state_dict(), save_path)

                curr_step += 1

        writer.close()

    # Test
    """
    elif args.mode == 'test':
        test_dataset = VQADataset(val_data, label_to_idx, args.val_img, max_sequence_length, word_idx_dicts, 
                                                                            Compose([ToTensor(), 
                                                                            Normalize((0.485, 0.456, 0.406), 
                                                                                      (0.229, 0.224, 0.225))]))
                              
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
