import os
import torch, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(1)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv6(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Train model
def train(model, device, train_loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        model_out = model(X_train)
        loss = criterion(model_out, y_train)
        total_loss += loss
        loss.backward()
        optimizer.step()
        y_pred = model_out.argmax(dim=1, keepdim=True)
        correct += y_pred.eq(y_train.view_as(y_pred)).sum().item()
    return correct / len(train_loader.dataset), total_loss/len(train_loader.dataset)

# Test model (try adaptive learning rate?)
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            model_out = model(X_test)
            pred = model_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_test.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('data'):
    os.makedirs('data')
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data/', train=True, download=True, transform=transform), 64)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data/', train=False, transform=transform), 256)

model = Net1().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

for epoch in range(1, 30):
    start = time.time()
    trn_acc, trn_loss = train(model, device, train_loader, optimizer)
    tst_acc = test(model, device, test_loader)
    scheduler.step()
    print('--------------\nEpoch:\t{}\nTrain:\t{:.4f}\nTest: \t{:.4f}\nTime:\t{:.4f}s'.format(epoch, trn_acc, tst_acc, time.time()-start))

#torch.save(model.state_dict(), "CNN_model.pt")
