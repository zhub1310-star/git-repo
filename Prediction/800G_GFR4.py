import os
import json
import onnxruntime as ort
import numpy as np

# ========= 路径配置 =========
model_dir = r"D:\\2xFR4_Prediction\\Model"
out_predict_path = r"D:\\AtsTempoData"
onnx_path = os.path.join(model_dir, "mlp_model.onnx")

# ========= 硬编码 scaler 参数 =========
scaler_X_mean = np.array(
    [148.18989290495315, 95.54270414993307, 1434.9670682730923, 1796.935341365462, 1.7745375913654629, 0.0], dtype=np.float32)
scaler_X_scale = np.array(
    [8.843122838759909, 6.261502159011312, 205.9775325015354, 7.857292845123487, 0.13911556434344505, 1.0], dtype=np.float32)

scaler_y_mean = np.array([148.18453815261043, 96.09082998661312, 1.7802908989959831, 2051.071686746988], dtype=np.float32)
scaler_y_scale = np.array([8.224885624027852, 6.2177884455125785, 0.14070758960302598, 9.408319766190449], dtype=np.float32)

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

# ========= 创建 ONNX 推理会话 =========
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# ========= 加载数据 =========
pred_file_path = r"D:\\AtsTempoData\\PredictedData.txt"
with open(pred_file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    header = [h.upper() for h in header]

data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32, usecols=range(0, 9))

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
#LowBias = outputs["LT"][:, 0]
#LowMod = outputs["LT"][:, 1]
#LowVEA = outputs["LT"][:, 2]
#LowTempAdc = outputs["LT"][:, 3]

# 平均温度 ADC 为整数
HighTempAdc[:] = int(np.mean(HighTempAdc))
#LowTempAdc[:] = int(np.mean(LowTempAdc))

results_array = np.column_stack([HighBias, HighMod, HighVEA, HighTempAdc])

# ========= 保存结果 =========
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savetxt(output_file, results_array,
           fmt="%.6f\t%.6f\t%.6f\t%d",
           header="HighBias\tHighMod\tHighVEA\tHighTEMPADC",
           comments="", encoding="utf-8")

print("✅ 预测完成，结果已保存：", output_file)
