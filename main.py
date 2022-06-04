'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from MyScheduler import myScheduler
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', default='ResNet50', type=str, help='VGG, ResNet, GoogLeNet, ResNext')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='CIFAR10, CIFAR100')
parser.add_argument('--scheduler', default='step', type=str, help='step, cyclic, one_cyclic, my1-4')
parser.add_argument('--gpu', default='0', type=str, help='different for different machines, 0-6 on my case')
parser.add_argument('--epoch', default=200, type=int, help='number of epoches to train')
args = parser.parse_args()

device = 'cuda:'+args.gpu if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
num_workers=4
data_dir = './data'
if args.dataset == 'CIFAR10':
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=num_workers)


elif args.dataset == 'CIFAR100':
    num_classes = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=num_workers)


# Model
print('==> Building model..')

# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net_name = args.net.lower()
if net_name == 'googlenet':
    net_name = 'GoogLeNet'
    print('GoogLeNet')
    net = GoogLeNet(num_classes=num_classes)
elif net_name == 'resnext':
    net_name = 'ResNeXt29'
    print('ResNeXt29_2x64d')
    net = ResNeXt29_2x64d(num_classes=num_classes)
elif net_name == 'densenet':
    net_name = 'DenseNet121'
    print('DenseNet121')
    net = DenseNet121(num_classes=num_classes)
elif net_name == 'vgg':
    net_name = 'VGG19'
    print('VGG19')
    net = VGG('VGG19', num_classes=num_classes)
else:
    net_name = 'ResNet50'
    print('ResNet50')
    net = ResNet50(num_classes=num_classes)
net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
if args.scheduler == 'step':
    print('step')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=782*90)
elif args.scheduler == 'cyclic':
    print('cyclic')
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=782*5, base_lr=0.001, max_lr=0.006)
elif args.scheduler == 'cosine':
    print('cosine')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=782*10, T_mult=2)
elif args.scheduler == 'my1':
    print('my1')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.1, peak_it=782*20, total_it=782*200)
elif args.scheduler == 'my2':
    print('my2')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.1, peak_it=782*50, total_it=782*200)
elif args.scheduler == 'my3':
    print('my3')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.01, peak_it=782*20, total_it=782*200)
elif args.scheduler == 'my4':
    print('my4')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.01, peak_it=782*50, total_it=782*200)
elif args.scheduler == 'my5':
    print('my5')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.01, peak_it=782*10, total_it=782*50)
elif args.scheduler == 'my6':
    print('my6')
    scheduler = myScheduler(optimizer, base_lr=args.lr, max_lr=0.1,
                            peak_it=782 * 10, total_it=782 * 50)
else:
    print('no scheduler')
    exit(1)

path = os.path.join('./results', '_'.join([args.dataset, net_name, args.scheduler]))
writer = SummaryWriter(path)
# Training
def train(epoch):
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
        total_ = targets.size(0)
        correct_ = predicted.eq(targets).sum().item()

        l = loss.item()
        acc = correct_/total_
        lr = optimizer.param_groups[0]['lr']
        step = batch_idx + epoch*782
        writer.add_scalars('Loss', {'train': l}, step)
        writer.add_scalars('Acc', {'train': acc}, step)
        writer.add_scalars('Err', {'train': 1.-acc}, step)
        writer.add_scalar('LR', lr, step)
        scheduler.step()
        total += total_
        correct += correct_
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))


def test(epoch):
    global best_acc
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = correct / total
    step = (epoch+1) * 782
    writer.add_scalars('Loss', {'test': test_loss/batch_idx}, step)
    writer.add_scalars('Acc', {'test': acc}, step)
    writer.add_scalars('Err', {'test': 1.-acc}, step)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(path, 'checkpoint.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)
