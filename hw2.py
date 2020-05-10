import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

SHOW_SAMPLES=False
EPOCHS = 10

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

if (SHOW_SAMPLES):
  plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

  figure = plt.figure()
  num_of_images = 60
  for index in range(1, num_of_images + 1):
      plt.subplot(6, 10, index)
      plt.axis('off')
      plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

plt.show()
# image size is 28x28
input_size = 784
def exractDigits(images,labels):
    zeroToSixIdx = np.where((labels>=0) & (labels<7))
    zeroToSixLabels = labels[zeroToSixIdx[0]]
    zeroToSixImages = images[zeroToSixIdx[0]]
    sevenToNineIdx = np.where(labels > 7)
    sevenToNineLabels = labels[sevenToNineIdx[0]]
    sevenToNineImages = images[sevenToNineIdx[0]]
    return [zeroToSixImages,zeroToSixLabels,sevenToNineImages,sevenToNineLabels]
im0,lbl0,im7,lbl7 = exractDigits(images.numpy(),labels.numpy())
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
        return F.log_softmax(x)


torch.manual_seed(42)
model = Net(D=input_size, H1=128, H2=64, Classes=10)
model.train()

criterion = nn.NLLLoss()

#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizer = optim.Adam(model.parameters())

time0 = time()
for e in range(EPOCHS):
    running_loss = 0
    for images, labels in trainloader:
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
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)



model.eval()
correct_count, all_count = 0, 0
for images,labels in valloader:
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
