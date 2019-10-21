import torch
import torch.nn as nn
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import VQADataset, ToTensor
from model import VQABaselineNet
from utils import sort_batch

"""
Run Train:

python3 main.py --mode train --model_name demo_model_vgg_scratch --root_dir /home/axe/Datasets/Scene_Context_Dataset 
--train_file img_triplets_train.txt --val_file img_triplets_val.txt  --log_dir /home/axe/Projects/MCS_VL/results_log/ 
--gpu_id 1 --num_epochs 50 --batch_size 64 --margin_triplet 2.0 
"""

"""
Run Test:
python3 main.py --mode test --model_name demo_model --root_dir /home/axe/Datasets/Scene_Context_Dataset 
--train_file img_triplets_train.txt --val_file img_triplets_val.txt  --log_dir /home/axe/Projects/MCS_VL/results_log/ 
--model_ckpt_file /home/axe/Projects/MCS_VL/results_log/demo_model/model_5000.ckpt --batch_size 64 --threshold_acc 0.2
"""


def str2bool(v):
    v = v.lower()
    assert v == 'true' or v == 'false'
    return v.lower() == 'true'


def compute_accuracy(model, dataloader, threshold, device, show_preds=False, mode='Validation'):
    """
    For the model, compute accuracy on triplet predictions
    :param model: ContextNet model
    :param dataloader: validation/test set dataloader
    :param threshold:
    :param device: cuda/cpu device where the model resides
    :return: None
    """
    model.eval()
    with torch.no_grad():
        num_correct = 0
        total = 0
        loss_val = 0

        # Evaluate on mini-batches
        for batch in dataloader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)

            # Compute correct predictions by comparing the L2 distances
            anchor_positive_l2_distance = torch.norm(anchor_emb - positive_emb, p=2, dim=1).cpu().numpy()
            anchor_negative_l2_distance = torch.norm(anchor_emb - negative_emb, p=2, dim=1).cpu().numpy()

            # |a - p| + t < |a - n|
            correct = np.less(anchor_positive_l2_distance + threshold, anchor_negative_l2_distance)

            num_correct += sum(correct)
            total += len(correct)
            loss_val += np.mean(anchor_positive_l2_distance - anchor_negative_l2_distance + threshold)

            # TODO: Visualize the inputs & display (save to disk) with correct choice (TensorBoardX)
            if show_preds:
                pass

        # Average loss across mini-batches
        loss_val = loss_val / total

        print('{} Accuracy: {:.2f} %'.format(mode, 100.0 * num_correct / total))
        print('{} Loss value: {:.4f}'.format(mode, loss_val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Context Model')

    parser.add_argument('--mode',             type=str,      help='train or test', default='train')
    parser.add_argument('--model_name',       type=str,      help='model save ckpt folder', default='baseline_model')
    parser.add_argument('--img_dir',          type=str,      help='path to images directory', required=True)
    parser.add_argument('--train_file',       type=str,      help='train file', required=True)
    parser.add_argument('--val_file',         type=str,      help='validation file')
    parser.add_argument('--batch_size',       type=int,      help='batch size', default=8)
    parser.add_argument('--num_epochs',       type=int,      help='number of epochs', default=50)
    parser.add_argument('--learning_rate',    type=float,    help='initial learning rate', default=1e-4)
    parser.add_argument('--num_gpus',         type=int,      help='number of GPUs to use for training', default=1)
    parser.add_argument('--gpu_id',           type=int,      help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=1)
    parser.add_argument('--log_dir',          type=str,      help='path to save model & summaries', required=True)
    parser.add_argument('--save_after',       type=int,      help='save model after every `n` weight update steps', default=3000)
    parser.add_argument('--vgg_wts_path',     type=str,      help='VGG-11 (bn) pre-trained weights (.pth) file')
    parser.add_argument('--threshold_acc',    type=float,    help='threshold margin for validation/test accuracy', default=0.2)
    parser.add_argument('--margin_triplet',   type=float,    help='margin value for triplet loss', default=2.0)
    parser.add_argument('--model_ckpt_file',  type=str,      help='path to saved model checkpoint file (.pth)')
    parser.add_argument('--is_vgg_pretrain',  type=str2bool, help='use pre-trained weights for VGG head', default=True)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Dataset directory
    root_dir = args.root_dir

    # Hyper params
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    # Load pre-trained weights for VGG-11 head
    if args.is_vgg_pretrain and args.vgg_wts_path:
        # from specified path
        vgg_11_pretrained_path = args.vgg_wts_path
    else:
        if args.is_vgg_pretrain:    # from default path
            vgg_11_pretrained_path = '/home/axe/Projects/Pre_Trained_Models/vgg11_bn-6002323d.pth'
        else:                       # no pre-trained wts
            vgg_11_pretrained_path = None

    # TODO: Multi-GPU PyTorch Implementation
    # if args.num_gpus > 1 and torch.cuda.device_count() > 1:
    #     print("Using {} GPUs!".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)

    # Train
    if args.mode == 'train':
        # Dataset & Dataloader
        train_file = os.path.join(root_dir, args.train_file)
        train_dataset = VQADataset(root_dir, train_file, transform=transforms.Compose([ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

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
            val_file = os.path.join(root_dir, args.val_file)
            val_dataset = VQADataset(root_dir, val_file, transform=transforms.Compose([ToTensor()]))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

        # LSTM params
        vocabulary_size = len(train_dataset.word2idx.keys())
        word_embedding_dim = 256
        lstm_hidden_units = 1024

        lstm_params = {'vocab_size': vocabulary_size, 'emb_dim': word_embedding_dim, 'enc_units': lstm_hidden_units}

        # Define model & load to device
        model = VQABaselineNet(lstm_params, vgg_11_pretrained_path, args.is_vgg_pretrain)
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

        # Save path
        save_dir = os.path.join(args.log_dir, args.model_name)
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

                # TODO:: Verify Sort Batch (based on seq length)
                question, label, ques_len = sort_batch(question, label, ques_len)

                # Load data onto the available device
                image = image.to(device)
                question = question.to(device)
                ques_len = ques_len.to(device)
                label = label.to(device)

                # Forward Pass
                label_predict = model(image, question)

                # Compute Loss
                loss = criterion(label_predict, label)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if (curr_step + 1) % 100 == 0:
                    # Validation set accuracy
                    if args.val_file:
                        compute_accuracy(model, val_loader, args.threshold_acc, device)

                    # Compute elasped & remaining time for training to complete
                    time_elapsed = (time.time() - start_time) / 3600
                    # total time = time_per_step * steps_per_epoch * total_epochs
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                            epoch + 1, n_epochs, curr_step + 1, steps_per_epoch, loss.item(), time_elapsed, time_left))

                # Save the target & reconstructed images to disk
                # if args.save_output and (curr_step + 1) % 500 == 0:
                #     plot_images_adjacent(img_target_batch[0].cpu().detach().numpy(),
                #                          img_reconstruct_batch[0].cpu().detach().numpy())

                # Save the model
                if (curr_step + 1) % args.save_after == 0:
                    print('Saving the model at the {} step to directory:{}'.format(curr_step + 1, save_dir))
                    save_path = os.path.join(save_dir, 'model_' + str(curr_step + 1) + '.pth')
                    torch.save(model.state_dict(), save_path)

                curr_step += 1

    # Test
    elif args.mode == 'test':
        val_file = os.path.join(root_dir, args.val_file)
        val_dataset = VQADataset(root_dir, val_file, transform=transforms.Compose([ToTensor()]))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

        checkpoint = torch.load(args.model_ckpt_file)
        # LSTM params
        vocabulary_size = -1     # TODO:: Retrieve from trained model (checkpoint file)
        word_embedding_dim = 256
        lstm_hidden_units = 1024

        lstm_params = {'vocab_size': vocabulary_size, 'emb_dim': word_embedding_dim, 'enc_units': lstm_hidden_units}

        # Define model & load to device
        model = VQABaselineNet(lstm_params, vgg_11_pretrained_path, args.is_vgg_pretrain)
        model.to(device)

        # Load pre-trained weights for validation
        model.load_state_dict(checkpoint)
        print('Model successfully loaded from {}'.format(args.model_ckpt_file))

        # Compute test accuracy
        compute_accuracy(model, val_loader, args.threshold_acc, device, show_preds=True, mode='Test')
