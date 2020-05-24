import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

SHOW_SAMPLES=False
LOAD_WEIGHTS = False
SAVE_WEIGHTS = False
EPOCHS = 8
PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Excercises/ex_2/'
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

# image size is 28x28
input_size = 784
def showSamples(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()
class Net(nn.Module):
    def __init__(self, D, H1, H2, Classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(D, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, Classes)
        self.bn1 = nn.BatchNorm1d(num_features=H1)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

def trainModel(model,trainloader,valloader):
    model.train()
    criterion = nn.NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    time0 = time()
    train_losses = list()
    test_losses = list()
    for e in range(EPOCHS):
        train_loss,test_loss = singleEpoch(model,optimizer,criterion,trainloader,valloader,e)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    return train_losses,test_losses
def singleEpoch(model,optimizer,criterion,trainSet,valSet,e):
    running_loss = 0
    for images, labels in trainSet:
        model.train()
            # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    val_batch_cnt = 0
    val_running_loss = 0
    model.eval()
    for val_batch_idx ,(val_images, val_labels) in enumerate(valSet):
            val_images = val_images.view(val_images.shape[0], -1)
            val_output = model(val_images)
            val_loss = criterion(val_output, val_labels)
            val_running_loss+= val_loss.item()
            val_batch_cnt+=1
            if(val_batch_idx==5):
                break
    print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainSet)))
    print("Test loss: {}".format(val_running_loss/val_batch_cnt))
    return running_loss/len(trainSet),val_running_loss/val_batch_cnt
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

def testModel(model,valSet):
    model.eval()
    correct_count, all_count = 0, 0
    for images,labels in valSet:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))
def saveWeights(model,filename):
    torch.save(model.state_dict(),PATH+filename)
def loadWeights(new_model,filename):
    new_model.load_state_dict(torch.load(PATH + filename))
    return new_model
def freezeFirstLayer(model):
    for name,param in model.named_parameters():
        if name in ['fc1.weight','fc1.bias']:
            param.requires_grad = False
        else:
            param.requires_grad = True
def initThirdLayer(model):
    for name,param in model.named_parameters():
        if name in ['fc3.weight']:
            nn.init.xavier_uniform_(param)
#showSamples(valloader_7t9)
torch.manual_seed(42)
model = Net(D=input_size, H1=128, H2=64, Classes=10)
if(LOAD_WEIGHTS):
    model = loadWeights(model,'weights_0t6.pth')
else:
    train_losses,test_losses = trainModel(model,trainloader_0t6,valloader_0t6)
    saveWeights(model,'weights_0t6.pth')
    x = np.arange(1,len(train_losses)+1)
    dtrainLosses = {'x':x,'y':train_losses,'funcName':'Train losses'}
    dtestLosses = {'x':x,'y':test_losses,'funcName':'Test losses'}
    #Multiplot([dtrainLosses,dtestLosses],'#Epochs','Accuracy',title='Loss')
if(SAVE_WEIGHTS):
    saveWeights(model,'weights_7t9.pth')
testModel(model,valloader_0t6)
print('freeze 1st layer. Initialize 3rd layer')
if(LOAD_WEIGHTS):
    loadWeights(model,filename='weights_7t9.pth')
else:
    freezeFirstLayer(model)
    initThirdLayer(model)
    train_losses,test_losses = trainModel(model,trainloader_7t9,valloader_7t9)
    #saveWeights(model,filename='weights_7t9.pth')
    x = np.arange(1,len(train_losses)+1)
    dtrainLosses = {'x':x,'y':train_losses,'funcName':'Train losses'}
    dtestLosses = {'x':x,'y':test_losses,'funcName':'Test losses'}
    #Multiplot([dtrainLosses,dtestLosses],'#Epochs','Accuracy',title='Seven to nine loss')
testModel(model,valloader_7t9)

print('new model')
model = Net(D=input_size, H1=128, H2=64, Classes=10)
if(LOAD_WEIGHTS):
    loadWeights(model,'new_weights_7t9.pth')    
else:
    train_losses,test_losses = trainModel(model,trainloader_7t9,valloader_7t9)
    #saveWeights(model,filename  ='new_weights_7t9.pth')
testModel(model,valloader_7t9)
