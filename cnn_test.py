
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np


num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

DATA_PATH = '..\data\MNIST'
MODEL_STORE_PATH = '..\models_for_test\\'


# transforms to apply to the data
#Compose() function. This function comes from the torchvision package. It allows the developer to setup various manipulations on the specified dataset. Numerous transforms can be chained together in a list using the Compose() function. In this case, first we specify a transform which converts the input data set to a PyTorch tensor
#he next argument in the Compose() list is a normalization transformation. Neural networks train better when the input data is normalized so that the data ranges from -1 to 1 or 0 to 1. To do this via the PyTorch Normalize transform, we need to supply the mean and standard deviation of the MNIST dataset, which in this case is 0.1307 and 0.3081 respectivel

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans,download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#test_dataset.data[0][27][27]
#..
#..
#.... test_dataset.data[1000][27][27]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1=  nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def forward(self, x):
        out = self.layer1(x)
        out2= self.layer2(out)
        #  reshaping function to out, which flattens the data dimensions from 7 x 7 x 64 into 3164 x 1
        out2 = out2.reshape(out2.size(0), -1)
        out3=self.drop_out(out2)
        fc1= self.fc1(out3)
        fc2 = self.fc2(fc1)
        return fc2



model =ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))




