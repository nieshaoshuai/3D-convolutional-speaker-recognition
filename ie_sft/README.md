# 信息抽取 SFT 训练代码（PyTorch）

该目录包含完整的四部分训练框架：

1. `dataset.py`：读取/生成数据，构建 batch。
2. `model.py`：加载开源基座模型（默认 `google/mt5-small`，参数量远小于 2B）。
3. `trainer.py`：训练主循环、每 10 step 评测、保存 Top-5 最优 checkpoint（循环覆盖）。
4. `eval.py`：评估精度/评测损失、保存日志、绘制精度与损失曲线。

## 数据格式
`ie_sft/data/sft_samples.json`：

```json
[
  {"input": "text", "output": "answer"}
]
```

若文件不存在，训练脚本会自动生成 100 条样例（抽取字段：`time/location/person/organization/title`）。

## 环境安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python ie_sft/train.py \
  --model_name google/mt5-small \
  --data_path ie_sft/data/sft_samples.json \
  --output_dir ie_sft/outputs \
  --batch_size 4 \
  --epochs 3 \
  --eval_every_steps 10
```

## 输出

- `ie_sft/outputs/metrics.json`：每次评测的指标日志。
- `ie_sft/outputs/precision_curve.png`：精度曲线图。
- `ie_sft/outputs/step_xxx/`：Top-5 最优 checkpoint（自动覆盖最差者）。


## 网页可视化（TensorBoard）

训练时会自动写入 TensorBoard 日志到：`ie_sft/outputs/tensorboard/`。

```bash
tensorboard --logdir ie_sft/outputs/tensorboard --port 6006
```

浏览器打开：`http://localhost:6006`，即可实时查看：

- `loss/train_step`：训练 step loss
- `loss/eval`：每 10 step 的评测 loss
- `metric/precision`：每 10 step 的评测精度

同时会生成静态图：

- `ie_sft/outputs/loss_curve.png`（训练损失 + 评测损失）
- `ie_sft/outputs/precision_curve.png`
