import torch
import torch.nn as nn
from code.mydataset import MyDataLoader
from code.mymodel import mymodel
from sklearn.metrics import r2_score


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    average_loss = total_loss / len(data_loader)
    r2 = r2_score(labels_list, predictions)

    print(f"Validation Loss: {average_loss:.4f}, R² Score: {r2:.4f}")
    return average_loss, r2

batch_size = 32
input_size = 16

val_file_path = r"D:\pycharm\cas_predict\calibration.xlsx"
val_loader = MyDataLoader(val_file_path, batch_size=batch_size, shuffle=False, num_workers=2)

lstm_model = mymodel(input_size)
lstm_model.load_state_dict(torch.load('../model/train_cas_predict2.pth'))

criterion = nn.MSELoss()

if __name__ == '__main__':
    val_loss, val_r2 = evaluate(lstm_model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, R² Score: {val_r2:.4f}")


