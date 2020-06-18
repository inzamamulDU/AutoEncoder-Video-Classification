import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas
import json
import os
import argparse

from model import CNNEncoder, RNNDecoder
from dataloader import Dataset
import config

def train_on_epochs(train_loader:DataLoader, test_loader:DataLoader, restore_from:str=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = nn.Sequential(
        CNNEncoder(**config.cnn_encoder_params),
        RNNDecoder(**config.rnn_decoder_params)
    )
    model.to(device)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('Use {} GPU training'.format(device_count))
        model = nn.DataParallel(model)

    ckpt = {}
    if restore_from is not None:
        ckpt = torch.load(restore_from)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % (restore_from))

    model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)

    if restore_from is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': []
    }

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0

    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for ep in range(start_ep, config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep, device)
        test_loss, test_score = validation(model, test_loader, optimizer, ep, device)

        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)

        ckpt_path = os.path.join(save_path, 'ep-%d.pth' % ep)
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': train_loader.dataset.labels
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))

    with open('./train_info.json', 'w') as f:
        json.dump(info, f)

    print('End of training')

def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()

def train(model:nn.Sequential, dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch, device):
    model.train()

    train_losses = []
    train_scores = []

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_ = model(X)

        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()

        y_ = y_.argmax(dim=1)
        acc = accuracy_score(y_.cpu().numpy(), y.cpu().numpy())

        train_losses.append(loss.item())
        train_scores.append(acc)

        if (i + 1) % config.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d: acc = %.2f, loss = %.2f' % (epoch, i + 1, len(dataloader), acc, loss.item()))

    return train_losses, train_scores

def validation(model:nn.Sequential, test_loader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch:int, device:int):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    test_loss = 0
    y_gd = []
    y_pred = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Validating'):
            X, y = X.to(device), y.to(device)
            y_ = model(X)

            loss = F.cross_entropy(y_, y, reduction='sum')
            test_loss += loss.item()

            y_ = y_.argmax(dim=1)
            y_gd += y.cpu().numpy().tolist()
            y_pred += y_.cpu().numpy().tolist()

    test_loss /= len(test_loader)
    test_acc = accuracy_score(y_gd, y_pred)

    print('[Epoch %3d]Test avg loss: %0.4f, acc: %0.2f\n' % (epoch, test_loss, test_acc))

    return test_loss, test_acc

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./data')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path

    dataloaders = {}
    for name in ['train', 'test']:
        raw_data = pandas.read_csv(os.path.join(data_path, '%s.csv' % name))
        dataloaders[name] = DataLoader(Dataset(raw_data.to_numpy()), **config.dataset_params)
    train_on_epochs(dataloaders['train'], dataloaders['test'], args.restore_from)
