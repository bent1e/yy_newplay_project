## 项目名称：yy新玩法AI识别（基于图像检索）

### 项目描述
本项目是一个基于Pytorch的yy新玩法AI识别项目，主要包括模型训练、模型推理功能。通过训练模型、验证准确性，以及应用到实际场景中进行AI识别，实现对玩法的自动识别和分析。
模型基于无标注的图像进行自监督训练（MAE tiny），训练结束后，提取模型encoder层的输入作为图像的特征，与检索库中的图像进行对比，判断主播是否进行游戏

### 项目文件结构
- `model.py`：MAE模型构建和特征提取网络实现
- `mae_pretrain.py`：模型训练脚本
- `second_val.py`：模型验证脚本
- `checkpoints/`：模型训练过程中的保存的检查点
- `gallery_data/`：用于玩法识别的检索库数据

### 使用说明
1. 如有需要，运行 `mae_pretrain.py` 对模型进行训练，模型结果将保存在checkpoints。
2. 模型推理请运行 `second_val.py` 对模型进行验证，检测模型对玩法的识别效果。


