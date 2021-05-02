# 사람의 손글씨 데이터인 MNIST를 이용해 Multi Layer Perceptron(MLP) 설계하기

### 설계 순서

1. 모듈 임포트하기
2. 딥러닝 모델을 설계할 때 활용하는 장비 확인하기
3. MNIST 데이터 다운로드 하기(Train set, Test set 분리)

4. 데이터 확인하기(1)

5. 데이터 확인하기(2)

6. MLP (Multi Layer Perceptron) 모델 설계하기
7. Optimizer, Objective Function 설정하기
8. MLP 모델 학습을 진행하면서 학습 데이터에 대한 모델 성능을 확인하는 함수 정의하기
9. 학습되는 과정 속에서 검증 데이터에 대한 모델의 성능을 확인하는 함수 정의하기
10. MLP 학습을 실행하면서 Train, Test set의 Loss 및 Test set Accuracy 확인하기



### 1. 모듈 임포트하기

``` python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
```

- numpy : 선행 대수와 관련된 함수 사용 모듈
- matplotlib.pyplot : 산출물 결과 시각화
- torch : pytorch 모듈
- torch.nn : 딥러닝 설계시 필요한 함수 모음
- torchvision : 컴퓨터 비전 연구 분야에서 자주 이용하는 모듈



### 2. 딥러닝 모델 설계할 때 활용하는 장비 확인

```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using pytorch version:', torch.__version__, 'Device: ', DEVICE)
```

딥러닝 모델 설계 혹은 딥러닝 모델을 구성하고 있는 파라미터 값을 업데이트 할 때 이용하는 장비를 선택한다. 

GPU를 이용한다면 계산 속도가 빠르기 때문에 파라미터의 값을 빠르게 업데이트할 수 있다. 



### 3. 하이퍼파라미터 지정

```python
BATCH_SIZE = 32
EPOCHS = 10
```

하이퍼 파라미터는 모델링할 때 사용자가 세팅해주는 값이다. learning rate나 SVM에서 C, sigma 값, KNN에서 K값 등이 있다. 정해진 최적의 값은 없지만 휴리스틱한 방법이나 경험 법칙에 의해 결정하는 경우가 있다. 베이지안 optimization 등으로 자동으로 하이퍼 파라미터를 선택해주는 라이브러리도 있긴하다. 

- `BATCH_SIZE = 32`

  MLP 모델을 학습할 때 필요한 데이터 개수의 단위이다. 

  학습 할 때 32개의 데이터를 이용해 첫 번째로 학습하고 그 다음 32개의 데이터를 이용해 두번째로 학습한다. 이 과정을 마지막 데이터까지 반복하여 학습이 진행된다. 

  `Iteration` : 한 개의 Mini-Batch를 이용해 학습하는 횟수이다. 32개의 데이터가 1개의 Mini-Batch가 된다.

  `Iteration` : 전체 데이터를 이용해 학습을 진행한 횟수

  Ex) 전체 데이터가 1만 개이고, 1000개 데이터를 이용해 1개의 Mini-Batch를 이용해 학습하는 횟수를 'iteration', 전체 데이터를 이용해 학습을 진행하는 횟수 'Epoch' 이다. 

- `EPOCHS = 10`

  존재하고 있는 Mni-batch를 전부 이용하는 횟수를 의미한다. 즉 전체 데이터셋을 10번 반복해 학습함을 의미한다. 



### 4. MNIST 데이터 다운로드 및 분리

``` python
train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST(root="../data/MNIST", 
                              train=False, 
                              transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

- `torchvision` 모듈 내 `datasets` 함수를 이용해 데이터셋을 다운로드한다. 

- datasets.MNIST

  - root

    데이터가 저장될 장소

  - train

    대상 데이터가 MLP 모델을 학습하기 위해 이용하는 학습용 데이터인지, 검증용 데이터인지 지정한다.

  - download

    해당 데이터를 인터넷 상에서 다운로드 해서 이용할 것인가.

  - transform

    데이터 다운로드 시 기본적인 전처리를 할 수 있다. 이 에제에서는 `ToTensor()` 메서드를 이용해 `tensor` 형태로 변경한다. 

- torch.utils.data.DataLoader

  - dataset 

    Mini-Batch 단위로 할당하고자 하는 데이터셋을 지정한다. DataLoader를 이용해 학습 진행 시 train_loader로 MLP 모델의 성능을 확인하는 용도로 test_loader로 설정한다.

  - batch_size

    Mini-batch 1개 단위를 구성하는 데이터의 개수를 지정한다. 

  - shuffle

    데이터의 순서를 섞고자 할 때 이용한다. MLP 모델이 학습 진행 시 특정 Label에 매칭된 이미지 데이터의 특징을 보고 학습하는 것이 아닌 특정 이미지 데이터에 매칭된 Label값을 집중적으로 학습하는 경우가 있다. 이때 잘못된 방향으로 학습하는 것을 방지하기 위해 데이터 순서를 섞는 과정을 진행한다.



### 5. 데이터 확인하기(1)

```python
for(x_train, y_train) in train_loader:
    print('X_train: ', x_train.size(), 'type: ', x_train.type())
    print('Y_train: ', y_train.size(), 'type: ', y_train.type())
    break
```

- x_train

  32개의 이미지 데이터가 1개의 Mini-Batch를 구성하고 있다. 이때 이미지 데이터는 가로 28개, 세로 28개의 픽셀로 구성되어 있으며 채널은 1인 Gray Scale로 이뤄진, 흑백 이미지 데이터이다.

- y_train

  32개의 이미지 데이터 각각에 label값이 1개씩 존재하므로 32개의 값을 갖는다. 



### 6. 데이터 확인하기(2)

```python
pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i,:,:,:].numpy().reshape(28, 28), cmap="gray_r")
    plt.title('Class: ' + str(y_train[i].item()))
```

다운로드 후 Mini-batch 단위로 할당된 데이터의 개수와 형태를 확인한다.



### 7. MLP(Multi Layer Perceptron) 모델 설계하기

``` python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
```

- `nn.Module`

  신경망을 작성하기 위해 nn.Module을 상속받는 Net 클래스를 정의한다. nn.Module 클래스가 이용할 수 있는 함수를 그대로 이용할 수 있으므로 새로운 딥러닝 모델 설계시 자주 사용된다.

- `self.fc1 = nn.Linear(28*28, 512)`

  첫 번째 Fully Conncected Layer를 정의한다. MNIST 데이터를 Input으로 사용하기 위해 28 x 28 x 1 (가로 픽셀 x 세로 픽셀 x 채널) 크기의 노드 수를 설정한 후 두번째 Fully Connected Layer의 노드 수를 512로 설정하였으므로 output 노드 수는 512가 된다.

- `self.fc2 = nn.Linear(512, 256)`

  두번째 Fully Connected Layer를 정의한다. 첫번째 Layer의 output 크기인 512 크기의 벡터 값을 Input으로 사용하기 위해 노드 수를 512개로 설정했고 세 번재 Fully Connected layer의 노드 수를 256으로 설정하므로 256이 output 노드 수가 된다.

- `self.fc3 = nn.Linear(256, 10) `

  최종적인 output으로 사용하기 위한 노드 수는 10개이다. 즉 결과를 0부터 9까지의 10가지 클래스로 표현하여야 한다. 이때 Label 값은 원-핫 인코딩으로 표현된다. 원-핫 인코딩이란 단 하나의 값만 True(Hot)이고 나머지는 모두 False(Cold)인 인코딩을 말한다. 

- `    def forward(self, x):`

  MLP 모델의 `Forward Propagation`을 정의한다. `Forward Propagation`이란 설계한 MLP 모델의 데이터를 입력했을 때 Output을 계산하기까지 과정을 나열한 것이다. 

- `        x = x.view(-1, 28*28)`

  이미지 데이터셋의 크기는 28x28로 2차원 데이터이므로 이를 1차원 데이터로 변환하기 위해 View 메서드를 이용해 784(=28x28) 크기의 1차원 데이터로 변환하여 진행한다. 즉 2차원의 데이터를 1차원으로 펼친다 라고 표현하며 `Flatten 한다` 라고 표현하기도 한다.

- `        x = self.fc1(x)`

  첫 번째 Fully Connected Layer에 1차원으로 펼쳐지 이미지 데이터를 통과시킨다.

- `        x = F.sigmoid(x)`

  `sigmoid()`를 이용해 두 번째 Fully Connected Layer의 Input으로 계산한다.

- `        x = self.fc2(x)`

  Fully Connected Layer에 `sigmoid()` 함수를 이용해 계산된 결괏값을 통과시킨다.

- `        x = F.sigmoid(x)`

  `sigmoid()` 를 이용해 세 번째 Fully Connected Layer의 Input으로 게산한다.

- `        x = self.fc3(x)`

  Fully Connected Layer에서 `sigmoid()` 함수를 이용해 계산된 결과값을 통과시킨다.

- `        x = F.log_softmax(x, dim=1)`

  최종 Output을 계산한다. 0부터 9까지, 총 10가지 경우의 수 중 하나로 분류하는 일을 수행해 softmax을 이용해 확률 값을 계산한다. 



### 8. Optimizer, Objective Function 설정하기

```python
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
```

- 기존에 설정한 device에 정의한 MLP 모델을 할당한다. 

- optimizer

  파라미터를 업데이트 할 때 이용하는 Optimizer를 정의한다.

  - learning rate : 알고리즘을 이용하여 파라미터를 업데이트할 때 반영
  - momentum : Optimizer의 관성

- `criterion = nn.CrossEntropyLoss()`

  Output 값과 계산될 Label 값은 class를 표현하는 원-핫 인코딩 값이다. MLP모델의 output 값과 원-핫 인코딩 값과의 Loss는 CrossEntropy를 이용해 계산하기 위해 사용한다. 

  **Cross Entropy에 대해 좀 더 알아보자.**
  $$
  H_{p}(q)=-\sum_{i=1}^{n} q\left(x_{i}\right) \log p\left(x_{i}\right)
  $$
  

  실제 분포를 q라고 하고, 모델링을 통하여 구한 분포를 p라고 할 때, p를 이용해 q를 예측하는 것을 말한다. 이때 q와 p가 모두 들어가서 크로스 엔트로피라고 한다. 

  실제 환경의 값과 예측값(관찰값)을 모두 알고 있는 경우에 ***머신러닝의 모델은 몇%의 확률로 예측했는데, 실제 확률은 몇%야*** 라는 사실을 알고 있을 때 사용한다. 

  실제와 예측값이 맞을 때 0으로 수렴하고, 값이 틀릴 경우 값이 커지므로, `실제 값과 예측 값의 차이를 줄이기 위한 엔트로피`이다. 



### 9. MLP 모델 학습 진행 시 학습 데이터에 대한 모델 성능 확인하는 함수

```python
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100 * batch_idx / len(train_loader),
                loss.item()))
```

- `optimizer.zero_grad()`

  기존에 정의한 장비에 이미지 데이터와 레이블 데이터를 할당한 경우, 과거에 이용한 Mini-Batch 내에 있는 이미지 데이터와 레이블 데이터를 바탕으로 계산된 Loss의 Gradient 값이 optimizer에 할당되어 있으므로 이를 초기화한다.

- `        loss.backward()`

  계산한 결과를 바탕으로 Back Propagation을 통해 계산된 Gradient 값을 각 파라미터에 할당한다.

- `optimizer.step()`

  파라미터 값을 업데이트한다.



### 10. 검증 데이터에 대한 모델 성능 확인하기

```python
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
```



### 11. 학습을 실행하면서 Train, Test set의 Loss 및 Test set Accuracy를 확인하기

```python
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy))
```



### 12. 전체 학습 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using pytorch version:', torch.__version__, 'Device: ', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST(root="../data/MNIST", 
                              train=False, 
                              transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


for(x_train, y_train) in train_loader:
    print('X_train: ', x_train.size(), 'type: ', x_train.type())
    print('Y_train: ', y_train.size(), 'type: ', y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i,:,:,:].numpy().reshape(28, 28), cmap="gray_r")
    plt.title('Class: ' + str(y_train[i].item()))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

겁

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy))
```



출처

- http://melonicedlatte.com/machinelearning/2019/12/20/204900.html
- 파이썬 딥러닝 파이토치