from __future__ import print_function, division

import torch
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import torch.nn as nn


plt.ion()  # interactive mode



data_dir = '/home/eli/vinson/images'

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train','val']}
train_dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=101,
                                               shuffle=True, num_workers=4)
                for x in ['train']}
val_dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=101,
                                               shuffle=False, num_workers=4)
                for x in ['val']}
dset_sizes = {x: len(dsets[x]) for x in ['train','val']}
print(dset_sizes)
dset_classes = dsets['train'].classes
print(dset_classes)

use_gpu = torch.cuda.is_available()
print(use_gpu)

def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(train_dset_loaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[dset_classes[x] for x in classes])
#plt.show()



def train_model(model, criterion, optimizer, scheduler, num_epoch=1):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc_1 = 0.0
    best_acc_5 = 0.0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects_1 = 0
            running_corrects_5 = 0
            if phase == 'train':
                scheduler.step()
                model.train(True) # Set model to training mode
                # Iterate over data
                for data in train_dset_loaders[phase]:
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable
                    if use_gpu:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    _, preds_top1 = torch.max(outputs.data, 1)
                    _, preds_top5 = torch.topk(outputs.data, 5, 1, largest=True)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects_1 += torch.sum(preds_top1 == labels.data)
                    for i in range(len(labels.data)):  # the number of labels
                        for j in range(3):  # number of topk
                            if preds_top5[i, j] == labels.data[i]:
                                running_corrects_5 = running_corrects_5 + 1
                            else:
                                j = j + 1
                        i = i + 1
            else:
                model.train(False) # Set model to evaluate mode
                # Iterate over data
                for data in val_dset_loaders[phase]:
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable
                    if use_gpu:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    _, preds_top1 = torch.max(outputs.data, 1)
                    _, preds_top5 = torch.topk(outputs.data, 5, 1, largest=True)

                    loss = criterion(outputs, labels)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects_5 += torch.sum(preds_top5 == labels.data)

                    for i in range(len(labels.data)):  # the number of labels
                        for j in range(5):  # number of topk
                            if preds_top5[i, j] == labels.data[i]:
                                running_corrects_5 = running_corrects_5 + 1
                            else:
                                j = j + 1
                        i = i + 1

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc_1 = running_corrects_1 / dset_sizes[phase]
            epoch_acc_5 = running_corrects_5 / dset_sizes[phase]

            print('{} Loss: {:.4f}  Top1_Accuracy: {:.4f}  Top5_Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_1, epoch_acc_5))
            # deep copy the model
            if phase == 'val' and epoch_acc_1 > best_acc_1 and epoch_acc_5 > best_acc_5:
                best_acc_1 = epoch_acc_1
                best_acc_5 = epoch_acc_5
                best_model_wts = model.state_dict()


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Top1_Accuracy: {:4f}".format(best_acc_1))
    print("Best val Top5_Accuracy: {:4f}".format(best_acc_5))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(val_dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

# Finetuning the convnet
model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 101)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoch=50)

visualize_model(model_ft)
plt.ioff()
plt.show()