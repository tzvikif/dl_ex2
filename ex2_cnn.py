import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
#learning_rate = 0.01
#momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
SHOW_SAMPLES=False
LOAD_WEIGHTS_MODEL1 = False
LOAD_WEIGHTS_MODEL2 = False
SAVE_WEIGHTS_MODEL1 = False
SAVE_WEIGHTS_MODEL2 = False
MODEL1_NAME = 'model1.pth'
MODEL2_NAME = 'model2.pth'
PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Excercises/ex_2/'
'''
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', train=True, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', train=False, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size_test, shuffle=True)
'''
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset_0t6 = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset_0t6 = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainset_0t6.data = trainset_0t6.data[trainset_0t6.targets < 7]
trainset_0t6.targets = trainset_0t6.targets[trainset_0t6.targets < 7]
valset_0t6.data = valset_0t6.data[valset_0t6.targets < 7]
valset_0t6.targets = valset_0t6.targets[valset_0t6.targets < 7]
trainloader_0t6 = torch.utils.data.DataLoader(trainset_0t6, batch_size=64, shuffle=True)
valloader_0t6 = torch.utils.data.DataLoader(valset_0t6, batch_size=64, shuffle=True)

trainset_7t9 = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset_7t9 = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainset_7t9.data = trainset_7t9.data[trainset_7t9.targets > 6]
trainset_7t9.targets = trainset_7t9.targets[trainset_7t9.targets > 6]
valset_7t9.data = valset_7t9.data[valset_7t9.targets > 6]
valset_7t9.targets = valset_7t9.targets[valset_7t9.targets > 6]
trainloader_7t9 = torch.utils.data.DataLoader(trainset_7t9, batch_size=64, shuffle=True)
valloader_7t9 = torch.utils.data.DataLoader(valset_7t9, batch_size=64, shuffle=True)


examples = enumerate(valloader_0t6)
batch_idx, (example_data, example_targets) = next(examples)
if(SHOW_SAMPLES):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
def Multiplot(l,xlabel,ylabel,title=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for p in l:
        x = p['x']
        y = p['y']
        funcName = p['funcName']
        plt.plot(x,y,label = funcName)
        plt.legend()
        plt.title(title)
        plt.plot()
    plt.show()
def freezeFirstLayer(model):
    for name,param in model.named_parameters():
        if name in ['conv1.weight','conv1.bias']:
            param.requires_grad = False
        else:
            param.requires_grad = True
def initThirdLayer(model):
    for name,param in model.named_parameters():
        if name in ['fc1.weight','fc2.weight']:
            nn.init.xavier_uniform_(param)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
def Main():
    network = Net()
    train_losses = []
    test_losses = []
    #optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(network.parameters())
    #test_counter = [i*len(trainloader_0t6.dataset) for i in range(n_epochs + 1)]
    print('0-6')
    if(not LOAD_WEIGHTS_MODEL1):
        test(network,valloader_0t6)
        for epoch in range(1, n_epochs + 1):
            train_loss = train(network,optimizer,epoch,trainloader_0t6,valloader_0t6)
            test_loss = test(network,valloader_0t6)
        saveWeights(network,PATH+MODEL1_NAME)
    else:
        network = loadWeights(network,PATH+MODEL1_NAME)
    test(network,valloader_0t6)
    print('7-9')
    if(not LOAD_WEIGHTS_MODEL2):
        freezeFirstLayer(network)
        initThirdLayer(network)
        test(network,valloader_7t9)
        for epoch in range(1, n_epochs + 1):
            train(network,optimizer,epoch,trainloader_7t9,valloader_7t9)
            test(network,valloader_7t9)
        saveWeights(network,PATH+MODEL2_NAME)
    else:
        network = loadWeights(network,PATH+MODEL2_NAME)
    test(network,valloader_7t9)
    print('7-9 new weights')
    network = Net()
    for epoch in range(1, n_epochs + 1):
        train(network,optimizer,epoch,trainloader_7t9,valloader_7t9)
        test(network,valloader_7t9)
    saveWeights(network,PATH+MODEL2_NAME)
    test(network,valloader_7t9)
def train(network,optimizer,epoch,trainloader,valloader):
    network.train()
    train_losses = []
    train_counter = []
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            train_losses.append(loss.item())
    return train_losses
      #train_counter.append(
      #  (batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))
      #torch.save(optimizer.state_dict(), 'optimizer.pth')
def test(network,testloader):
    network.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in testloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]  #shape=[13,1]
            correct += torch.sum(pred.eq(target.data.view_as(pred)),dim=0)
        test_loss /= len(testloader.dataset)
        test_losses.append(test_loss)
        print(f'\nTest set: Avg. loss: {test_loss}, Accuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset)}%')
    return test_losses
def loadWeights(new_model,path):
    new_model.load_state_dict(torch.load(path))
    return new_model
def saveWeights(model,path):
    torch.save(model.state_dict(),path)
#Start point
Main()