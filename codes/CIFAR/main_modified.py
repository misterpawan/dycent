'''Train CIFAR with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from myoptims.Ours import Ours
from torch.optim import lr_scheduler
import os
import argparse
from torchvision import datasets, models
from models import *


import sys
sys.path.append('../')


import random

dataset = 'cifar10' # Change to cifar100 to test the script on cifar100 dataset
epochs = 100
model = None
bs = 128
resume = 0
alg = None
lr = None
beta = None
manualSeed  = None

def get_loaders(dsetname, bsize):
    print('==> Preparing ' + dsetname + ' data...')
    if dsetname == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        torchdset = torchvision.datasets.CIFAR10
    elif dsetname == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        torchdset = torchvision.datasets.CIFAR100
    else:
        print('==> Dataset not avaiable...')
        exit()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchdset(root='/scratch/'+dsetname+'/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=4,drop_last=True)
    testset = torchdset(root='/scratch/'+dsetname+'/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    return trainloader, testloader

inputs, targets = None, None
net = None

def get_model(modelname, Num_classes):
    global net
    if   modelname == 'v16':  net = VGG('VGG16',    Num_classes=Num_classes)
    elif modelname == 'r18':  net = ResNet18(       Num_classes=Num_classes)
    elif modelname == 'r34':  net = ResNet34(       Num_classes=Num_classes)
    elif modelname == 'r50':  net = ResNet50(       Num_classes=Num_classes)
    elif modelname == 'r101': net = ResNet101(      Num_classes=Num_classes)
    elif modelname == 'rx29': net = ResNeXt29_4x64d(Num_classes=Num_classes)
    elif modelname == 'dla':  net = DLA(            Num_classes=Num_classes)
    elif modelname == 'd121': net = DenseNet121(    Num_classes=Num_classes)
    else:
        print('==> Network not found...')
        exit()
    return net
train = None
test = None

def get_optim(optim_name, learning_rate, net):
    global alpha, beta, train, test
    if   optim_name == 'sgd':
      optimizer = optim.SGD(     net.parameters(), lr=learning_rate, momentum=0.9)
      train = train_reg
      test = test_reg
    elif optim_name == 'rmsprop':
      optimizer = optim.RMSprop( net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'adam':
      optimizer = optim.Adam(    net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'adamw':
      optimizer = optim.AdamW(   net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'diffgrad':
      optimizer = diffgrad(      net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'adabelief':
      optimizer = AdaBelief(     net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'cosangulargrad':
      optimizer = cosangulargrad(net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'tanangulargrad':
      optimizer = tanangulargrad(net.parameters(), lr=learning_rate)
      train = train_reg
      test = test_reg
    elif optim_name == 'Ours':
      #net.parameters = list(net.parameters())
      optimizer = Ours(params= list(net.parameters()), steps=1, h=learning_rate,alpha=-1, beta=beta,loss_func=loss_func,plr=learning_rate)
      train = train_ours
      test = test_ours

    else:
        print('==> Optimizer not found...')
        exit()
    return optimizer
'''
dataset = 'cifar10'
epochs = 100
model = 'r50'
bs = 128
resume = 0
alg = 'Ours'
lr = 1e-2
'''
import csv
z = 0
csv_file = None
csv_file_p1_u = None
csv_file_p2_u = None
def write_rows(accuracy = None):
  global z
  if(accuracy == None):
    global writer
    global csv_file
    csv_file = open(  dataset + "_" + model + "_" + alg  +"_"  + '_'+str(z)+  '_accuracy.csv', 'w')
    writer = csv.DictWriter(csv_file, fieldnames = ['accuracy'])
    writer.writeheader()
    csv_file.flush()

  else:
    dict_data = {'accuracy' : str(accuracy),
                      }
    writer.writerow(dict_data)
    csv_file.flush()
csv_file_p1_u = None
csv_file_p2_u = None
loss_csv = None
writer_loss_train = None
write_loss_test = None
train_loss_csv = None
def write_rows_loss_train(loss = None):
  global z
  if(loss == None):
    global writer_loss_train
    global train_loss_csv
    train_loss_csv = open(  dataset + "_" + model + "_" + alg    + '_'+str(z)+  '_loss_train.csv', 'w')
    writer_loss_train = csv.DictWriter(train_loss_csv, fieldnames = ['loss'])
    writer_loss_train.writeheader()
    train_loss_csv.flush()

  else:
    dict_data = {'loss' : str(loss),
                      }
    writer_loss_train.writerow(dict_data)
    train_loss_csv.flush()
test_loss_csv = None
train_loss_csv = None

def write_rows_loss_test(loss = None):
  global z
  if(loss == None):
    global writer_loss_test
    global test_loss_csv
    test_loss_csv = open(  dataset + "_" + model + "_" + alg  +"_"  + '_'+str(z)+  '_loss_test.csv', 'w')
    writer_loss_test = csv.DictWriter(test_loss_csv, fieldnames = ['loss'])
    writer_loss_test.writeheader()
    test_loss_csv.flush()

  else:
    dict_data = {'loss' : str(loss),
                      }
    writer_loss_test.writerow(dict_data)
    test_loss_csv.flush()
def write_rows_angle_p1_u(angle = None):
  global z
  if(angle == None):
    global writer_p1
    global csv_file_p1_u
    csv_file_p1_u = open(  dataset + "_" + model + "_" + alg  +"_"  + '_'+str(z)+  '_p1_u.csv', 'w')
    writer_p1 = csv.DictWriter(csv_file_p1_u, fieldnames = ['angle'])
    writer_p1.writeheader()
    csv_file_p1_u.flush()

  else:
    dict_data = {'angle' : str(angle),
                      }
    writer_p1.writerow(dict_data)
    csv_file_p1_u.flush()
    
def write_rows_angle_p2_u(angle = None):
  global z
  if(angle == None):
    global writer_p2
    global csv_file_p2_u
    csv_file_p2_u = open(  dataset + "_" + model + "_" + alg  +"_"  + '_'+str(z)+  '_p2_u.csv', 'w')
    writer_p2 = csv.DictWriter(csv_file_p2_u, fieldnames = ['angle'])
    writer_p2.writeheader()
    csv_file_p2_u.flush()

  else:
    dict_data = {'angle' : str(angle),
                      }
    writer_p2.writerow(dict_data)
    csv_file_p2_u.flush()


def loss_func():
  global criterion, net, inputs, targets, debug

  outputs = net(inputs)
  loss = criterion(outputs, targets)
  return loss

def train_ours(trainloader, epoch, net, optimizer, criterion, device='cuda'):
    global inputs, targets
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #optimizer.zero_grad()
        loss = loss_func()
        #loss.backward()
        optimizer.step(loss)

        train_loss += loss.item()
        outputs = net(inputs).detach()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #optimizer.avg_p1_p2 = self.angle(self.p1_grad, self.p2_grad)
        #print(" AVG : ", optimizer.avg_p1_p2.item()* ( 180 / torch.pi)," CURRENT : ", optimizer.theta.item()* ( 180 / torch.pi))
    print('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),correct/total))
    acc=100.*correct/total
    write_rows_loss_train(train_loss/(batch_idx+1))

    write_rows_angle_p1_u(optimizer.theta.item() * 180/torch.pi)
    write_rows_angle_p2_u(optimizer.avg_angle.item() * 180/torch.pi)
    optimizer.iter = 0
    
    return acc, train_loss/(batch_idx+1)


def test_ours(testloader, epoch, net, criterion, device='cuda'):
    global inputs, targets
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = loss_func()

            test_loss += loss.detach().item()
            outputs = net(inputs).detach()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Testing:  Loss: {:.4f} | Acc: {:.4f}'.format(test_loss/(batch_idx+1),correct/total) )
    acc=100.*correct/total
    write_rows_loss_test(test_loss/(batch_idx+1))
    write_rows(acc)
    return acc, test_loss/(batch_idx+1)






def train_reg(trainloader, epoch, net, optimizer, criterion, device='cuda'):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),correct/total))
    acc=100.*correct/total
    return acc, train_loss/(batch_idx+1)


def test_reg(testloader, epoch, net, criterion, device='cuda'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Testing:  Loss: {:.4f} | Acc: {:.4f}'.format(test_loss/(batch_idx+1),correct/total) )
    acc=100.*correct/total
    write_rows(acc)
    return acc, test_loss/(batch_idx+1)



optimizer = None

def main():
    global model,criterion, net, inputs, targets, debug, optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global manualSeed
    # Random seed
    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(manualSeed)

    trainloader, testloader = get_loaders(dataset, bs)
    #print(model)
    net = get_model(model, 10 if dataset == 'cifar10' else 100)

    if device == 'cuda':
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt' + '_' + dataset + '_' + model + '.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = -1
        start_epoch = 0


    optimizer = get_optim(alg, lr, net)
    criterion = nn.CrossEntropyLoss()
    scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)


    for epoch in range(start_epoch, start_epoch+epochs):
        train_acc, train_loss = train(trainloader, epoch, net, optimizer, criterion, device=device)
        scheduler_lr.step()
        val_acc, val_loss = test(testloader, epoch, net, criterion, device=device)

        # Save checkpoint.
        if val_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            #if not os.path.isdir('checkpoint'): Uncomment to start saving checkpoints
            #    os.mkdir('checkpoint')
            #torch.save(state, './checkpoint/ckpt' + '_' + dataset + '_' + model + '.t7')
            best_acc = val_acc

    print('Best Acc: {:.2f}'.format(best_acc))
    del net







'''
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', type=str, default='cifar10', \
                            choices=['cifar10', 'cifar100'], \
                            help='dataset (options: cifar10, cifar100)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--model', type=str, default='r50', \
                            choices=['v16', 'r18', 'r34', 'r50', 'r101', 'rx29', 'dla', 'd121'], \
                            help='dataset (options: v16, r18, r34, r50, r101, rx29, dla, d121)')
parser.add_argument('--bs', default=128, type=int, help='batchsize')
parser.add_argument('--alg', type=str, default='adam', \
                            choices=['sgd', 'rmsprop', 'adam', 'adamw', 'diffgrad', 'adabelief', 'cosangulargrad', 'tanangulargrad'], \
                            help='dataset (options: sgd, rmsprop, adam, adamw, diffgrad, adabelief, cosangulargrad, tanangulargrad)')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--manualSeed', default=1111, type=int, help='random seed')

args = parser.parse_args()
'''
hyperparameter = {
    "sgd" : 1e-2,
    "rmsprop" : 1e-3,
    "adam" : 1e-3,
    "diffgrad" : 1e-3,
    "adabelief" : 1e-3,
    "cosangulargrad" : 1e-3,
    "tanangulargrad" : 1e-2,
    "Ours" : 1e-2,

}

architectures = ["r18", "r34",]
betas = [0.2,0.2, 0.2, 0.2, 0.2, 0.2]

for architecture_num, architecture in enumerate(architectures):
  for key in hyperparameter.keys():
    model = architectures[architecture_num]
    alg = key
    lr = hyperparameter[key]
    beta = betas[architecture_num]
    for z in range(1):
      write_rows()
      write_rows_angle_p1_u()
      write_rows_angle_p2_u()
      write_rows_loss_train()
      write_rows_loss_test()
      main()
      csv_file.close()
      csv_file_p1_u.close()
      csv_file_p2_u.close()
      train_loss_csv.close()
      test_loss_csv.close()
