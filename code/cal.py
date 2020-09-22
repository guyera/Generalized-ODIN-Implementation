# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import argparse
import os

from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from resnet import ResNet34
from wideresnet import WideResNet
from deconfnet import DeconfNet, CosineDeconf, InnerDeconf, EuclideanDeconf

from generatingloaders import Normalizer, GaussianLoader, UniformLoader

from tqdm import tqdm

r_mean = 125.3/255
g_mean = 123.0/255
b_mean = 113.9/255
r_std = 63.0/255
g_std = 62.1
b_std = 66.7

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])


h_dict = {
    'cosine':   CosineDeconf,
    'inner':    InnerDeconf,
    'euclid':   EuclideanDeconf
}



def kliep_loss(logits, labels, max_ratio=50):
    softplus = nn.Softplus()
    logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
    
    #preds  = torch.softmax(logits,dim=1)
    preds  = softplus(logits)
    #preds  = torch.sigmoid(logits) * 10

    maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
    
    y = torch.eye(preds.size(1))
    labels = y[labels].to(preds.device)

    inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
    outlier_loss = ((1-labels) * (preds)).mean(1)
    loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

    return loss

losses_dict = {
    'ce':nn.CrossEntropyLoss(),
    'kliep': kliep_loss, 
}

generating_loaders_dict = {
    'Gaussian': GaussianLoader,
    'Uniform': UniformLoader
}

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
    # Device arguments
    parser.add_argument('--gpu', default = 0, type = int,
                        help = 'gpu index')

    # Model loading arguments
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--model-dir', default = './models', type = str,
                        help = 'model name for saving')

    # Architecture arguments
    parser.add_argument('--architecture', default = 'densenet', type = str,
                        help = 'underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument('--similarity', default = 'cosine', type = str,
                        help = 'similarity function for decomposed confidence numerator (cosine | inner | euclid | baseline)')
    parser.add_argument('--loss-type', default = 'ce', type = str,
                        help = 'ce|kliep')

    # Data loading arguments
    parser.add_argument('--data-dir', default='./data', type = str)
    parser.add_argument('--out-dataset', default = 'Imagenet', type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--batch-size', default = 64, type = int,
                        help = 'batch size')

    # Training arguments
    parser.add_argument('--no-train', action='store_false', dest='train')
    parser.add_argument('--weight-decay', default = 0.0001, type = float,
                        help = 'weight decay during training')
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'number of epochs during training')

    # Testing arguments
    parser.add_argument('--no-test', action='store_false', dest='test')
    parser.add_argument('--magnitudes', nargs = '+', default = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                        help = 'perturbation magnitudes')
    
    
    parser.set_defaults(argument=True)
    return parser.parse_args()

def get_datasets(data_dir, data_name, batch_size):

    train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transform)
    test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transform)
    
    if data_name == 'Gaussian' or data_name == 'Uniform':
        normalizer = Normalizer(r_mean, g_mean, b_mean, r_std, g_std, b_std)
        outlier_loader = generating_loaders_dict[data_name](batch_size = batch_size, num_batches = int(10000 / batch_size), transformers = [normalizer])
    else:
        outlier_set  = torchvision.datasets.ImageFolder(f'{data_dir}/{data_name}', transform=test_transform)
        outlier_loader       =  DataLoader(outlier_set,       batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_indices      = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in       = Subset(test_set_in, test_indices[1000:])

    train_loader_in      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
    validation_loader_in =  DataLoader(validation_set_in, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader


def main():
    args = get_args()
    
    device           = args.gpu
    
    load_model       = args.load_model
    model_dir        = args.model_dir

    architecture     = args.architecture
    similarity       = args.similarity
    loss_type        = args.loss_type
    
    data_dir         = args.data_dir
    data_name        = args.out_dataset
    batch_size       = args.batch_size
    
    train            = args.train
    weight_decay     = args.weight_decay
    epochs           = args.epochs

    test             = args.test
    noise_magnitudes = args.magnitudes

    # Create necessary directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if architecture == 'densenet':
        underlying_net = DenseNet3(depth = 100, num_classes = 10)
    elif architecture == 'resnet':
        underlying_net = ResNet34()
    elif architecture == 'wideresnet':
        underlying_net = WideResNet(depth = 28, num_classes = 10, widen_factor = 10)
    
    underlying_net.to(device)
    
    # Construct g, h, and the composed deconf net
    baseline = (similarity == 'baseline')
    
    if baseline:
        h = InnerDeconf(underlying_net.output_size, 10)
    else:
        h = h_dict[similarity](underlying_net.output_size, 10)

    h.to(device)

    deconf_net = DeconfNet(underlying_net, underlying_net.output_size, 10, h, baseline)
    
    deconf_net.to(device)

    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == 'h.h.weight' or name == 'h.h.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr = 0.1, momentum = 0.9, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    h_optimizer = optim.SGD(h_parameters, lr = 0.1, momentum = 0.9) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    # Load the model (capable of resuming training or inference)
    # from the checkpoint file

    if load_model:
        checkpoint = torch.load(f'{model_dir}/checkpoint.pth')
        
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        h_optimizer.load_state_dict(checkpoint['h_optimizer'])
        deconf_net.load_state_dict(checkpoint['deconf_net'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        h_scheduler.load_state_dict(checkpoint['h_scheduler'])
        epoch_loss = checkpoint['epoch_loss']
    else:
        epoch_start = 0
        epoch_loss = None

    #get outlier data
    train_data, val_data, test_data, open_data = get_datasets(data_dir, data_name, batch_size)
  
    criterion = losses_dict[loss_type]

    # Train the model
    if train:
        deconf_net.train()
        
        num_batches = len(train_data)
        epoch_bar = tqdm(total = num_batches * epochs, initial = num_batches * epoch_start)
        
        for epoch in range(epoch_start, epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_data):
                if epoch_loss is None:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{num_batches}')
                else:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {batch_idx + 1}/{num_batches}')
                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()
                
                logits, _, _ = deconf_net(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                
                optimizer.step()
                h_optimizer.step()
                total_loss += loss.item()
                
                epoch_bar.update()
            
            epoch_loss = total_loss
            h_scheduler.step()
            scheduler.step()
            
            checkpoint = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'h_optimizer': h_optimizer.state_dict(),
                'deconf_net': deconf_net.state_dict(),
                'scheduler': scheduler.state_dict(),
                'h_scheduler': h_scheduler.state_dict(),
                'epoch_loss': epoch_loss,
            }
            torch.save(checkpoint, f'{model_dir}/checkpoint.pth') # For continuing training or inference
            torch.save(deconf_net.state_dict(), f'{model_dir}/model.pth') # For exporting / sharing / inference only
        
        if epoch_loss is None:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Batch {num_batches}/{num_batches}')
        else:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {num_batches}/{num_batches}')
        epoch_bar.close()

    if test:
        deconf_net.eval()
        best_val_score = None
        best_auc = None
        
        for score_func in ['h', 'g', 'logit']:
            print(f'Score function: {score_func}')
            for noise_magnitude in noise_magnitudes:
                print(f'Noise magnitude {noise_magnitude:.5f}         ')
                validation_results =  np.average(testData(deconf_net, device, val_data, noise_magnitude, criterion, title = 'Validating'))
                print('ID Validation Score:',validation_results)
                
                id_test_results = testData(deconf_net, device, test_data, noise_magnitude, criterion, title = 'Testing ID') 
                
                ood_test_results = testData(deconf_net, device, open_data, noise_magnitude, criterion, title = 'Testing OOD')
                auroc = calc_auroc(id_test_results, ood_test_results)*100
                tnrATtpr95 = calc_tnr(id_test_results, ood_test_results)
                print('AUROC:', auroc, 'TNR@TPR95:', tnrATtpr95)
                if best_auc is None:
                    best_auc = auroc
                else:
                    best_auc = max(best_auc, auroc)
                if best_val_score is None or validation_results > best_val_score:
                    best_val_score = validation_results
                    best_val_auc = auroc
                    best_tnr = tnrATtpr95
        
        print('supposedly best auc: ', best_val_auc, ' and tnr@tpr95 ', best_tnr)
        print('true best auc:'      , best_auc)

def calc_tnr(id_test_results, ood_test_results):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=.95)]

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result

def testData(model, CUDA_DEVICE, data_loader, noise_magnitude, criterion, score_func = 'h', title = 'Testing'):
    model.eval()
    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)
    for j, (images, _) in enumerate(data_iter):
        data_iter.set_description(f'{title} | Processing image batch {j + 1}/{num_batches}')
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        
        if score_func == 'h':
            _, scores, _ = model(images)
        elif score_func == 'g':
            _, _, scores = model(images)
        elif score_func == 'logit':
            scores, _, _ = model(images)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of the numerator w.r.t. input

        max_scores, _ = torch.max(scores, dim = 1)
        max_scores.backward(torch.ones(len(max_scores)).to(CUDA_DEVICE))
        
        # Normalizing the gradient to binary in {-1, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
        gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
        gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)
        
        # Now calculate score
        if score_func == 'h':
            _, scores, _ = model(images)
        elif score_func == 'g':
            _, _, scores = model(images)
        elif score_func == 'logit':
            scores, _, _ = model(images)

        results.extend(torch.max(scores, dim=1)[0].data.cpu().numpy())
        
    data_iter.set_description(f'{title} | Processing image batch {num_batches}/{num_batches}')
    data_iter.close()

    return np.array(results)

if __name__ == '__main__':
    main()
