import torch
from torch import optim, nn
import pandas as pd
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_csv = pd.read_csv('./data.csv', usecols=[1])
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))


def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


X, Y = create_dataset(dataset)

train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
train_X = X[:train_size]
train_Y = Y[:train_size]
test_X = X[train_size:]
test_Y = Y[train_size:]

train_x = torch.from_numpy(train_X).to(device)
train_y = torch.from_numpy(train_Y).to(device)
test_x = torch.from_numpy(test_X).to(device)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=1, sequence_length=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = LSTM(10).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(5000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)

    out = model(var_x)
    loss = criterion(out, var_y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.data.item()))


model=model.eval()

X=X.reshape(-1,1,10)
X=torch.from_numpy(X).to(device)