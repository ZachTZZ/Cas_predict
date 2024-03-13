import torch
from mydataset import MyDataLoader
from mymodel import mymodel
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

def scheduler(epoch, lr):
    if epoch >15 :
        return lr
    if epoch % 6 == 0 and epoch > 0:
        return lr * 0.8
    else:
        return lr

batch_size = 64
learning_rate = 0.1
num_epochs = 60

train_file_path = r"D:\pycharm\cas_predict\train.xlsx"
train_loader = MyDataLoader(train_file_path, batch_size=batch_size, shuffle=True, num_workers=2)

input_size = 16

lstm_model = mymodel(input_size)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
scheduler_lr = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler(epoch, learning_rate))
criterion = nn.MSELoss()

def train(model, train_loader, optimizer, num_epochs=10):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler_lr.step()

        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    plt.plot(losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('r-loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'train_cas_predict3.pth')

if __name__ == '__main__':
    train(lstm_model, train_loader, optimizer, num_epochs)


