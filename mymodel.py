import torch
import torch.nn as nn
import torch.optim as optim

class mymodel(nn.Module):
    def __init__(self, input_size):
        super(mymodel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=4, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(4)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(8)
        self.dense1 = nn.Linear(8, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.dense2 = nn.Linear(32, 16)
        self.batch_norm4 = nn.BatchNorm1d(16)
        self.dense3 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batch_norm1(x[:, -1, :])
        x, _ = self.lstm2(x.unsqueeze(1))
        x = self.batch_norm2(x[:, -1, :])
        x = torch.relu(self.batch_norm3(self.dense1(x)))
        x = torch.relu(self.batch_norm4(self.dense2(x)))
        x = self.dense3(x)
        return x


# input_size = 34
# hidden_size = 64
#
#
# lstm_model = CustomModel(input_size, hidden_size)
#
# # 打印模型结构
# print(lstm_model)




