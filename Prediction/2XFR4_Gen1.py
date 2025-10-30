import os
import json
import onnxruntime as ort
import numpy as np

# ========= 路径配置 =========
model_dir = r"D:\\2xFR4_Prediction\\Model"
out_predict_path = r"D:\\AtsTempoData"
onnx_path = os.path.join(model_dir, "mlp_model.onnx")
params_path = os.path.join(model_dir, "params.json")

# ========= 硬编码 scaler 参数 =========
scaler_X_mean = np.array(
    [174.75952946322565, 43.43918327813535, 1824.1082765731426, 1074.8068174717241, 0.5943879205461475,
     0.5001405986377554], dtype=np.float32)
scaler_X_scale = np.array(
    [14.051241153193153, 3.8793114156563457, 246.49709001674947, 35.5470370606701, 0.09952687321288638,
     0.49999998023202263], dtype=np.float32)

scaler_y_mean = np.array([169.8787258639005, 42.99351684059239, 0.5946010304474161, 1108.710241829657], dtype=np.float32)
scaler_y_scale = np.array([15.765209445432745, 4.108937874171759, 0.10181346516638187, 137.41316764038692], dtype=np.float32)

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

# ========= 读取参数文件（可选） =========
with open(params_path, "r") as f:
    params = json.load(f)

# ========= 创建 ONNX 推理会话 =========
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# ========= 加载数据 =========
pred_file_path = r"D:\\AtsTempoData\\PredictedData.txt"
with open(pred_file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    header = [h.upper() for h in header]

data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32, usecols=range(0, 10))

# 找到所需列索引
col_idx = {name: i for i, name in enumerate(header)}
feature_cols = ["RT_BIASDAC", "RT_MODDAC", "RT_IEAADC", "RT_TEMPADC", "RT_VEA(MV)"]

# ========= ONNX 推理 =========
outputs = {}
for key, val in zip(["LT", "HT"], [0, 1]):
    feats = np.column_stack([data[:, col_idx[col]] for col in feature_cols])
    feats = np.column_stack([feats, np.full((feats.shape[0],), val, dtype=np.float32)])
    X_scaled = scale_X(feats)
    ort_inputs = {"input": X_scaled}
    y_pred_scaled = ort_session.run(None, ort_inputs)[0]
    outputs[key] = inverse_scale_y(y_pred_scaled)

# ========= 构造结果矩阵 =========
HighBias = outputs["HT"][:, 0]
HighMod = outputs["HT"][:, 1]
HighVEA = outputs["HT"][:, 2]
HighTempAdc = outputs["HT"][:, 3]
LowBias = outputs["LT"][:, 0]
LowMod = outputs["LT"][:, 1]
LowVEA = outputs["LT"][:, 2]
LowTempAdc = outputs["LT"][:, 3]

# 平均温度 ADC 为整数
HighTempAdc[:] = int(np.mean(HighTempAdc))
LowTempAdc[:] = int(np.mean(LowTempAdc))

results_array = np.column_stack([HighBias, HighMod, HighVEA, HighTempAdc,
                                 LowBias, LowMod, LowVEA, LowTempAdc])

# ========= 保存结果 =========
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savetxt(output_file, results_array,
           fmt="%.6f\t%.6f\t%.6f\t%d\t%.6f\t%.6f\t%.6f\t%d",
           header="HighBias\tHighMod\tHighVEA\tHighTEMPADC\tLowBias\tLowMod\tLowVEA\tLowTEMPADC",
           comments="", encoding="utf-8")

print("✅ 预测完成，结果已保存：", output_file)
