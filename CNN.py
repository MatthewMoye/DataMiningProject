import torch, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8*8*128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
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
    print('Train: ', correct / len(train_loader.dataset))
    return correct / len(train_loader.dataset), total_loss/len(train_loader.dataset)

# Test model
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            model_out = model(X_test)
            pred = model_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_test.view_as(pred)).sum().item()
    print('Test: ', correct / len(test_loader.dataset))
    return correct / len(test_loader.dataset)


use_cuda = torch.cuda.is_available()
torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

# Change range of values to [0,1]
trnsfrm = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True, transform=trnsfrm), 64)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, transform=trnsfrm), 256)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(1, 20):
    start = time.time()
    print('------------\nepoch: ', epoch)
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)
    scheduler.step()
    print(time.time()-start, 's\n')

#torch.save(model.state_dict(), "CNN_model.pt")
