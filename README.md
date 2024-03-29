## 前言
本项目中的代码是专门为参加大连理工大学（DUT）交叉学科竞赛的生物工程（生工）赛道而开发的。
本项目的成功完成是我学习过程中的一个小里程碑，因为是对一个学期里学到的深度学习的内容的一次应用。（同时也借此学了git和github的基本操作嘿嘿）
下面的内容是GPT4基于代码生成的，基本准确的概括了该项目的内容。

## 概览

该项目分为几个关键组成部分，包括数据加载与预处理、模型定义、训练、评估和预测模块。每个组件对于处理DNA序列数据、训练LSTM模型、评估其性能以及对新的未见数据进行预测都至关重要。

### 数据加载与预处理

- **DNADataset 类**：一个自定义的数据集类，用于从Excel文件加载DNA序列及其相应的分数。它包括用于将DNA序列进行独热编码和k-mer嵌入的方法，以将序列表示为适合LSTM模型的固定大小输入向量。
- **MyDataLoader 函数**：一个实用函数，将`DNADataset`与`DataLoader`封装起来，用于高效的批处理、随机排序和并行数据加载。

### 模型定义

- **mymodel 类**：定义了神经网络架构。模型由两个LSTM层组成，用于序列处理，然后是批量归一化和密集层，用于预测分数。该模型架构旨在捕捉DNA的序列特性，并学习与分数相关的复杂模式。

### 训练

- **train 函数**：处理模型训练过程。它通过批次迭代训练数据集，使用均方误差（MSE）准则计算损失，执行反向传播，并更新模型的权重。学习率调度器在训练期间调整学习率以改善收敛性。

### 评估

- **evaluate 函数**：评估模型在验证或测试数据集上的性能。它计算MSE损失和R²分数，提供关于模型预测准确性的洞见以及其捕捉数据方差的能力。

### 预测

- **predict 函数**：为新的、未见过的DNA序列生成预测。它通过训练好的模型处理这些序列并输出预测分数，然后将结果保存到Excel文件中以供进一步分析。

### 使用方法

要使用DNA序列预测模型，请按照以下步骤操作：

1. 准备您的DNA序列数据，格式为Excel，包含序列和分数的列。
2. 调整`file_path`变量以指向您的训练和验证/测试数据集。
3. 根据需要自定义`batch_size`、`num_workers`和`kmer_length`。
4. 通过执行`train`函数并设定所需的训练周期数来运行训练阶段。
5. 使用`evaluate`函数在单独的数据集上评估模型性能。
6. 使用`predict`函数为新的DNA序列预测分数，并保存结果。





