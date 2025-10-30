import os
import json
import numpy as np
import onnxruntime as ort

# ========= 硬编码 scaler 参数 =========
scaler_X_mean = np.array([988.5794065688431, 257.04135270577206, 565.6523776904496,
                          1074.781014529329, 0.44740459684553263, 0.5000180264628474], dtype=np.float32)
scaler_X_scale = np.array([142.81301018072693, 78.2923671910123, 86.54396088951111,
                           35.19777192128263, 0.08874468458075417, 0.49999999967504666], dtype=np.float32)

scaler_y_mean = np.array([1338.389371597505, 1108.6812921368569, 0.4439448333744554], dtype=np.float32)
scaler_y_scale = np.array([414.0544492599595, 124.00234670670207, 0.09259104104401727], dtype=np.float32)

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

# ========= 路径设置 =========
model_dir = r"D:\\dll\\FR4_SiPh_Prediction\\Model"
out_predict_path = r"D:\\AtsTempoData"
onnx_path = os.path.join(model_dir, "mlp_model.onnx")


# ========= 创建 ONNX 推理会话 =========
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# ========= 加载数据 =========
pred_file_path = r"D:\\AtsTempoData\\PredictedData.txt"

# 先读取第一行获取列名
with open(pred_file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    header = [h.upper() for h in header]

# 列名映射
rename_map = {
    "BIASDAC": "RT_BIASDAC",
    "MPDADC": "RT_MPDADC",
    "PDADC": "RT_PDADC",
    "TEMPADC": "RT_TEMPADC",
    "RATIO": "RT_RATIO"
}
header = [rename_map.get(col, col) for col in header]

# 把数据读进 numpy
data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32)

# 找到所需列索引
col_idx = {name: i for i, name in enumerate(header)}
feature_cols = ["RT_BIASDAC", "RT_MPDADC", "RT_PDADC", "RT_TEMPADC", "RT_RATIO"]

# ========= 构造 LT、HT 数据 =========
outputs = {}
for key, val in zip(["LT", "HT"], [0, 1]):
    # 在最后一列添加状态
    feats = np.column_stack([data[:, col_idx[col]] for col in feature_cols])
    feats = np.column_stack([feats, np.full((feats.shape[0],), val, dtype=np.float32)])

    # 标准化
    X_scaled = scale_X(feats)

    # 推理
    ort_inputs = {"input": X_scaled}
    y_pred_scaled = ort_session.run(None, ort_inputs)[0]

    # 反标准化
    outputs[key] = inverse_scale_y(y_pred_scaled)

# ========= 构造结果 =========
HighBias = outputs["HT"][:, 0]
HighRatio = outputs["HT"][:, 2]
HighTempAdc = outputs["HT"][:, 1]
LowBias = outputs["LT"][:, 0]
LowRatio = outputs["LT"][:, 2]
LowTempAdc = outputs["LT"][:, 1]


# 平均温度ADC为整数
HighTempAdc[:] = int(np.mean(HighTempAdc))
LowTempAdc[:] = int(np.mean(LowTempAdc))

# ========= 判断条件: TXLOP > 2.1 时，LowBias - 140 =========
if "TXLOP_DCA(DBM)" in col_idx:
    txlop = data[:, col_idx["TXLOP_DCA(DBM)"]]   # 取出每行的 TXLOP 值
    for i in [0, 1]:  # 只检查第1、2行（索引 0、1）
        if txlop[i] > 2.1:
            LowBias[i] -= 140
    for i in [0, 1, 2, 3]:  # 只检查第1、2行（索引 0、1）
        if txlop[i] > 2.1:
            HighBias[i] -= 140

# 拼成二维数组
results = np.column_stack([HighBias, HighRatio, HighTempAdc, LowBias, LowRatio, LowTempAdc])

# 保存
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savetxt(output_file,
           results,
           fmt="%.6f\t%.6f\t%d\t%.6f\t%.6f\t%d",
           header="HighBias\tHighRatio\tHighTEMPADC\tLowBias\tLowRatio\tLowTEMPADC",
           comments="",
           encoding="utf-8")

print("✅ 预测完成，结果已保存：", output_file)
