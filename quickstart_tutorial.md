```python
%matplotlib inline
```

Quickstart
===================
This section runs through the API for common tasks in machine learning.

Working with data
-----------------
PyTorch has two `primitives to work with data `:
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.  
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.





```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```

In this tutorial, we
use the FashionMNIST dataset.  
 Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
``target_transform`` to modify the samples and labels respectively.




```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/26421880 [00:00<?, ?it/s]


    Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/29515 [00:00<?, ?it/s]


    Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/4422102 [00:00<?, ?it/s]


    Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/5148 [00:00<?, ?it/s]


    Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    


We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.




```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```

    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64


--------------




Creating Models
------------------
To define a neural network in PyTorch, we create a class that inherits
from `nn.Module`.  
We define the layers of the network
in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function.  
 To accelerate
operations in the neural network, we move it to the GPU if available.




```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

    Using cuda device
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )


Read more about `building neural networks in PyTorch `.




Optimizing the Model Parameters
----------------------------------------
To train a model, we need a `loss function`
and an `optimizer `.




```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
backpropagates the prediction error to adjust the model's parameters.




```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also check the model's performance against the test dataset to ensure it is learning.




```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
accuracy increase and the loss decrease with every epoch.




```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 2.306208  [    0/60000]
    loss: 2.297800  [ 6400/60000]
    loss: 2.276234  [12800/60000]
    loss: 2.267652  [19200/60000]
    loss: 2.259894  [25600/60000]
    loss: 2.230015  [32000/60000]
    loss: 2.232131  [38400/60000]
    loss: 2.197122  [44800/60000]
    loss: 2.190398  [51200/60000]
    loss: 2.166457  [57600/60000]
    Test Error: 
     Accuracy: 45.5%, Avg loss: 2.155203 
    
    Epoch 2
    -------------------------------
    loss: 2.160062  [    0/60000]
    loss: 2.154296  [ 6400/60000]
    loss: 2.092179  [12800/60000]
    loss: 2.108840  [19200/60000]
    loss: 2.062736  [25600/60000]
    loss: 2.004668  [32000/60000]
    loss: 2.027419  [38400/60000]
    loss: 1.942268  [44800/60000]
    loss: 1.942519  [51200/60000]
    loss: 1.885460  [57600/60000]
    Test Error: 
     Accuracy: 59.8%, Avg loss: 1.872615 
    
    Epoch 3
    -------------------------------
    loss: 1.894987  [    0/60000]
    loss: 1.870397  [ 6400/60000]
    loss: 1.746129  [12800/60000]
    loss: 1.797140  [19200/60000]
    loss: 1.698699  [25600/60000]
    loss: 1.639545  [32000/60000]
    loss: 1.670323  [38400/60000]
    loss: 1.558378  [44800/60000]
    loss: 1.583873  [51200/60000]
    loss: 1.494495  [57600/60000]
    Test Error: 
     Accuracy: 62.3%, Avg loss: 1.503062 
    
    Epoch 4
    -------------------------------
    loss: 1.557461  [    0/60000]
    loss: 1.529375  [ 6400/60000]
    loss: 1.376835  [12800/60000]
    loss: 1.459445  [19200/60000]
    loss: 1.355081  [25600/60000]
    loss: 1.334155  [32000/60000]
    loss: 1.362995  [38400/60000]
    loss: 1.275227  [44800/60000]
    loss: 1.312025  [51200/60000]
    loss: 1.226740  [57600/60000]
    Test Error: 
     Accuracy: 63.9%, Avg loss: 1.244510 
    
    Epoch 5
    -------------------------------
    loss: 1.309896  [    0/60000]
    loss: 1.295827  [ 6400/60000]
    loss: 1.132673  [12800/60000]
    loss: 1.241669  [19200/60000]
    loss: 1.130763  [25600/60000]
    loss: 1.138601  [32000/60000]
    loss: 1.174589  [38400/60000]
    loss: 1.100914  [44800/60000]
    loss: 1.140659  [51200/60000]
    loss: 1.070969  [57600/60000]
    Test Error: 
     Accuracy: 65.0%, Avg loss: 1.082498 
    
    Done!


Read more about `Training your model `.




Saving Models
-------------
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).




```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

    Saved PyTorch Model State to model.pth


Loading Models
----------------------------

The process for loading a model includes re-creating the model structure and loading
the state dictionary into it.




```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>



This model can now be used to make predictions.




```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

    Predicted: "Ankle boot", Actual: "Ankle boot"


Read more about `Saving & Loading your model `.



