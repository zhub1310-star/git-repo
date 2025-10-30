import os
import sys
import numpy as np
import onnxruntime as ort

# ========= 硬编码 scaler 参数 =========
scaler_X_mean = np.array([990.7686477475382, 334.0998875707529, 519.6195433046445, 1814.2920252771962, 
                          0.6465749689052099, 638.6500606057823, 0.4982747925874234], dtype=np.float32)
scaler_X_scale = np.array([143.72059706917838, 54.34087028172857, 87.7313098946925, 7.10134388992778, 
                           0.058966341976823294, 98.02210435072496, 0.49999702365052495], dtype=np.float32)
scaler_y_mean = np.array([868.7782611514053, 1878.192292781267, 0.6492790193785677], dtype=np.float32)
scaler_y_scale = np.array([277.45874404050994, 216.2488853500238, 0.061159621202690584], dtype=np.float32)

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

out_predict_path = r"D:\\AtsTempoData"
onnx_path = os.path.join(base_dir, "mlp_model.onnx")

ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

pred_file_path = r"D:\\AtsTempoData\\PredictedData.txt"

with open(pred_file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    header = [h.upper() for h in header]

rename_map = {
    "BIASDAC": "RT_BIASDAC",
    "MPDADC": "RT_MPDADC",
    "PDADC": "RT_PDADC",
    "TEMPADC": "RT_TEMPADC",
    "RATIO": "RT_RATIO"
}
header = [rename_map.get(col, col) for col in header]

data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32)

# === 计算 RT_BIASRATIO ===
bias_col = header.index("RT_BIASDAC")
ratio_col = header.index("RT_RATIO")
bias_ratio = (data[:, bias_col] * data[:, ratio_col]).reshape(-1, 1)

# === 拼接新列到 data ===
data = np.hstack([data, bias_ratio])

# === 更新 header ===
header.append("RT_BIASRATIO")

col_idx = {name: i for i, name in enumerate(header)}
feature_cols = ["RT_BIASDAC", "RT_MPDADC", "RT_PDADC", "RT_TEMPADC", "RT_RATIO", "RT_BIASRATIO"]

outputs = {}
for key, val in zip(["LT", "HT"], [0, 1]):

    feats = np.column_stack([data[:, col_idx[col]] for col in feature_cols])
    feats = np.column_stack([feats, np.full((feats.shape[0],), val, dtype=np.float32)])

    X_scaled = scale_X(feats)

    ort_inputs = {"input": X_scaled}
    y_pred_scaled = ort_session.run(None, ort_inputs)[0]

    outputs[key] = inverse_scale_y(y_pred_scaled)

HighBias = outputs["HT"][:, 0]
HighRatio = outputs["HT"][:, 2]
HighTempAdc = outputs["HT"][:, 1]
LowBias = outputs["LT"][:, 0]
LowRatio = outputs["LT"][:, 2]
LowTempAdc = outputs["LT"][:, 1]

HighTempAdc[:] = int(np.mean(HighTempAdc))
LowTempAdc[:] = int(np.mean(LowTempAdc))

RT_Ratio = data[:, col_idx["RT_RATIO"]]
HighBias = HighBias / RT_Ratio
LowBias = LowBias / RT_Ratio

results = np.column_stack([LowBias, HighBias, LowTempAdc, HighTempAdc, LowRatio, HighRatio])

bias_cols = [0, 3]  # HighBias,  LowBias
pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]  # 按行配对
# 偏置和PD列做平均（按配对）
for col in bias_cols:  # 遍历偏置列
    for i, j in pairs:
        avg = int(np.mean([results[i, col], results[j, col]]))
        results[i, col] = results[j, col] = avg

# 保存
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savetxt(output_file,
           results,
           fmt="%d\t%d\t%d\t%d\t%.6f\t%.6f",
           header="LowBias\tHighBias\tLowTEMPADC\tHighTEMPADC\tLowRatio\tHighRatio",
           comments="",
           encoding="utf-8")

print("✅ 预测完成，结果已保存：", output_file)
