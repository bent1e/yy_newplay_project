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

### 基本使用说明
- 1. 环境配置`requirements.txt`
- 2. 如有需要，运行 `mae_pretrain.py` 对模型进行训练，模型结果将保存在checkpoints。
- 3. 模型推理请运行 `second_val.py` 对模型进行验证，检测模型对玩法的识别效果。

### 模型推理说明
- 1. `--batch_size` 对于检索库图中提取的图像批次数，默认=100
- 2. `--val_batch_size` 对于待识别图中提取的图像批次数，默认=1
- 3. `--model_path` 模型文件位置，默认`./checkpoints/epoch_48.pth`
- 4. `--gallery_path`检索库位置
- 5. `--test_path` 待检索图像文件描述，是一个txt，每行代表图像地址，如/data1/zhuzhipeng/yy_newplay/data/small_new_game/1354946443_2712473029_2b4e954a-8271-4b6f-a481-502e7542f36b.jpg
* 模型先对于检索库的图像进行特征提取，并保存为中间变量，再对每一张待识别样例进行特征提取并检索（二次检索），生成待识别图像的label（1=玩法，0=非玩法）
