import torch
import pandas as pd
from code.mydataset import MyDataLoader
from code.mymodel import mymodel


def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())

    return predictions


if __name__ == '__main__':
    test_file_path = r"D:\pycharm\cas9\test.xlsx"
    batch_size = 32
    input_size = 16

    # 创建测试数据的DataLoader
    test_loader = MyDataLoader(test_file_path, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    lstm_model = mymodel(input_size)

    # 加载训练好的模型参数
    lstm_model.load_state_dict(torch.load('../model/train_cas_predict2.pth'))

    # 进行预测
    predictions = predict(lstm_model, test_loader)

    # 创建包含预测结果的DataFrame
    df = pd.DataFrame({"scores": predictions})

    # 将DataFrame保存到Excel文件
    path = r"D:\pycharm\cas_predict\小组4r.xlsx"
    df.to_excel(path, index=False)

    print(f"预测结果已保存至 {path}")

