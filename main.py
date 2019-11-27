import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataloader import VQADataset
from torch.optim.lr_scheduler import StepLR
from utils import sort_batch, load_vocab
from utils import str2bool, int_min_two, plot_data, print_and_log

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
    parser.add_argument('--mode',          type=str,            help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir',      type=str,            help='root directory to save model & summaries', required=True)
    parser.add_argument('--expt_name',     type=str,            help='expt_dir/expt_name: organize experiments', required=True)
    parser.add_argument('--run_name',      type=str,            help='expt_dir/expt_name/run_name: organize training runs', required=True)

    # Data params
    parser.add_argument('--train_img',     type=str,            help='path to training images directory', required=True)
    parser.add_argument('--train_file',    type=str,            help='training dataset file', required=True)
    parser.add_argument('--val_img',       type=str,            help='path to validation images directory')
    parser.add_argument('--val_file',      type=str,            help='validation dataset file')
    parser.add_argument('--num_cls', '-K', type=int_min_two,    help='top K answers (labels); min = 2', default=1000)

    # Vocab params
    parser.add_argument('--vocab_file',    type=str,            help='vocabulary pickle file (gen. by prepare_data.py)')

    # Training params
    parser.add_argument('--batch_size',    '-bs',   type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',    '-ep',   type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate', '-lr',   type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--log_interval',  type=int,            help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval', type=int,            help='save model after `n` weight update steps', default=3000)
    parser.add_argument('--val_size',      type=int,            help='validation set size for evaluating accuracy', default=10000)

    # Evaluation params
    parser.add_argument('--K_eval',        type=int,            help='top-K labels during evaluation/inference', default=1000)

    # Model params
    parser.add_argument('--model_ckpt',    type=str,            help='resume training/perform inference; e.g. model_1000.pth')
    parser.add_argument('--vgg_wts_path',  type=str,            help='VGG-11 (bn) pre-trained weights (.pth) file', default=PATH_VGG_WEIGHTS)
    parser.add_argument('--vgg_train',     type=str2bool,       help='whether to train the VGG encoder', default='false')
    # parser.add_argument('--model_config', type=str, help='model config file - specifies model architecture')

    # GPU params
    # parser.add_argument('--num_gpus',   type=int,   help='number of GPUs to use for training', default=1)
    parser.add_argument('--gpu_id',        type=int,            help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=0)
    parser.add_argument('--opt_lvl',       type=int,            help='Automatic-Mixed Precision: opt-level (O_)', default=1, choices=[0, 1, 2, 3])

    # Misc params
    parser.add_argument('--num_workers',   type=int,            help='number of worker threads for Dataloader', default=1)

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
        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Load vocab (.pickle) file
        vocab = load_vocab(args.vocab_file)
        print('Vocabulary loaded from {}'.format(args.vocab_file))

        # Unpack vocab
        word2idx, idx2word, label2idx, idx2label, max_seq_length = [v for k, v in vocab.items()]

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logs_file(parser, log_dir)

        # Dataset & Dataloader
        train_dataset = VQADataset(args.train_file, args.train_img, word2idx, label2idx, max_seq_length,
                                   transform=Compose([Resize((224, 224)), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                                                (0.229, 0.224, 0.225))]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                                   drop_last=True, num_workers=args.num_workers)

        print('Question Vocabulary Size: {} \n\n'.format(len(word2idx.keys())))

        print('Train Data Size: {}'.format(train_dataset.__len__()))

        # Plot data (image, question, answer) for sanity check
        # plot_data(train_loader, idx2word, idx2label, num_plots=80)
        # sys.exit()

        if args.val_file:
            # Use the same word-index dicts as that obtained for the training set
            val_dataset = VQADataset(args.val_file, args.val_img, word2idx, label2idx, max_seq_length,
                                     transform=Compose([Resize((224, 224)), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                                                  (0.229, 0.224, 0.225))]))

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True,
                                                     drop_last=True, num_workers=args.num_workers)

            log_msg = 'Validation Data Size: {}\n'.format(val_dataset.__len__())
            log_msg += 'Accuracy on the validation set is computed using {} samples. See --val_size\n'.format(args.val_size)

            print_and_log(log_msg, log_file)

        # Question Encoder params
        vocabulary_size = len(train_dataset.word2idx.keys())
        word_embedding_dim = 300
        encoder_hidden_units = 1024

        question_encoder_params = {'vocab_size': vocabulary_size, 'inp_emb_dim': word_embedding_dim,
                                   'enc_units': encoder_hidden_units, 'batch_size': batch_size}

        # Image Encoder params
        is_vgg_trainable = args.vgg_train  # default = False
        vgg_wts_path = args.vgg_wts_path  # default = PATH_VGG_WTS

        image_encoder_params = {'is_trainable': is_vgg_trainable, 'weights_path': vgg_wts_path}

        # Num of classes is the top-K labels + 1 (for UNKNOWN)
        num_classes = args.num_cls + 1

        # Define model & load to device
        model = VQABaselineNet(question_encoder_params, image_encoder_params, K=num_classes)
        model.to(device)

        # Load model checkpoint file (if specified) from `log_dir`
        if args.model_ckpt:
            model_ckpt_path = os.path.join(log_dir, args.model_ckpt)
            checkpoint = torch.load(model_ckpt_path)

            model.load_state_dict(checkpoint)

            log_msg = 'Model successfully loaded from {}'.format(model_ckpt_path) + '\nResuming Training...'

            print_and_log(log_msg, log_file)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)

        # TODO: StepLR Scheduler
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O{}".format(args.opt_lvl))

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
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if (curr_step + 1) % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if args.val_file:
                        validation_metrics = compute_validation_metrics(model, val_loader, device, size=args.val_size)

                        # Reset the mode to training
                        model.train()

                        log_msg = 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}'.format(
                                validation_metrics['accuracy'], validation_metrics['loss'])

                        print_and_log(log_msg, log_file)

                        # If current model has the best accuracy on the validation set & >= training accuracy,
                        # save model to disk

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Accuracy', validation_metrics['accuracy'], curr_step)
                        writer.add_scalar('Val/Loss', validation_metrics['loss'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    # total time = time_per_step * steps_per_epoch * total_epochs
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                        epoch + 1, n_epochs, curr_step + 1, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_and_log(log_msg, log_file)

                # Save the model
                if (curr_step + 1) % args.save_interval == 0:
                    print('Saving the model at the {} step to directory:{}'.format(curr_step + 1, log_dir))
                    save_path = os.path.join(log_dir, 'model_' + str(curr_step + 1) + '.pth')
                    torch.save(model.state_dict(), save_path)

                curr_step += 1

            # Validation set accuracy on the entire set
            if args.val_file:
                # Total validation set size
                total_validation_size = val_dataset.__len__()
                validation_metrics = compute_validation_metrics(model, val_loader, device, total_validation_size)

                log_msg = '\nAfter {} epoch:\n'.format(epoch+1)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    validation_metrics['accuracy'], validation_metrics['loss'])

                print_and_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

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


def compute_validation_metrics(model, dataloader, device, size):
    """
    For the given model, computes accuracy & loss on validation/test set.

    :param model: VQA model
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: metrics {'accuracy', 'loss'}
    :rtype: dict
    """
    model.eval()
    with torch.no_grad():
        batch_size = dataloader.batch_size
        loss = 0.0
        num_correct = 0

        n_iters = size // batch_size

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

            # Compute Loss
            loss += F.cross_entropy(label_logits, label, reduction='mean')

            if i >= n_iters:
                break

        # Total Samples
        total = n_iters * batch_size

        # Final Accuracy
        accuracy = 100.0 * num_correct / total

        # Final Loss (averaged over mini-batches - n_iters)
        loss = loss / n_iters

        metrics = {'accuracy': accuracy, 'loss': loss}

        return metrics


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


if __name__ == '__main__':
    main()