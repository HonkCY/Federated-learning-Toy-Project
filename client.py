import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
# pytorch staff above
import pickle
import gzip
import requests
import time
from io import BytesIO

#%%%%%%%%%%%%%%%%%%%%%
# data processing
transform = transforms.Compose([
    transforms.ToTensor(),  # 將圖片轉成Tensor
    transforms.Normalize((0.5,), (0.5,))  # 標準化
])

# get data
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

#%%%%%%%%%%%%%%%%%%%%%

client_id = "client1"

# model

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# functions
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model_to_gzip(model, filepath):
    parameters = model.state_dict()
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(parameters, f)

def load_model_from_gzip(filepath):
    with gzip.open(filepath, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def restore_model_from_gzip(model, filepath):
    parameters = load_model_from_gzip(filepath)
    model.load_state_dict(parameters)
    return model

def get_avg_model_from_server(filepath,epoch):
    url = 'http://127.0.0.1:5000/model?epoch='+str(epoch)
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False

def retry_get_avg_model_from_server(t, max_retries=10, retry_interval=30):
    for attempt in range(max_retries):
        if get_avg_model_from_server(f"{t}.pkl.gz",t):
            return True
        time.sleep(retry_interval)
    return False

def upload_parameters_to_server(model):
    global client_id
    url = f"http://127.0.0.1:5000/upload?client_id={client_id}"
    memfile = BytesIO()
    with gzip.GzipFile(fileobj=memfile, mode='wb') as f:
        pickle.dump(model.state_dict(), f)
    memfile.seek(0)
    files = {'model': memfile}
    response = requests.post(url, files=files)
    return response.status_code == 200

model = SimpleNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    
    # 上傳模型參數到server
    if upload_parameters_to_server(model):
        print("Parameters uploaded successfully.")
    else:
        print("Failed to upload parameters.")

    # 嘗試從server獲取平均後的模型
    if retry_get_avg_model_from_server(t):
        restore_model_from_gzip(model, f"{t}.pkl.gz")
    else:
        print("fail")
    test(test_loader, model, loss_fn)
    
print("Done!")
