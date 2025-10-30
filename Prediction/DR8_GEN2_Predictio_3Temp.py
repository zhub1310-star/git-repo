import os
import sys
import numpy as np 
import onnxruntime as ort

# ========= 硬编码 scaler 参数 =========
scaler_X_mean = np.array(
    [1555.9086458333334, 6.035620312500041, 256.8657565712683, 526.1287435226151, 1247.5511094741175, 
     0.42198203124997047, 1876.5686197916666, 0.9391826430208304, 1171.070703125, 1.0001822916666667], dtype=np.float32)
scaler_X_scale = np.array(
    [351.2432322720487, 0.5783372543730255, 25.032114471335117, 34.202728500891055, 66.84281221111645, 
     0.021279298186047547, 10.356430654792486, 0.020999997013695502, 58.33427448072606, 0.8165125076220536], dtype=np.float32)

scaler_y_mean = np.array([1555.9086458333334, 0.4496650974612206, 1880.0865885416667], dtype=np.float32)
scaler_y_scale = np.array([351.2432322720487, 0.023476980815478065, 173.69543740808552], dtype=np.float32)

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

data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32)

col_idx = {name: i for i, name in enumerate(header)}
bias_pwr = data[:, col_idx["BIASDAC"]] / data[:, col_idx["MAXPOWER"]]
data = np.column_stack([data, bias_pwr])
header.append("BIASPWR")
col_idx["BIASPWR"] = len(header) - 1

feature_cols = ["BIASDAC", "MAXPOWER", "BIASPWR", "REALMPDADC", "REALPDADC", "REALRATIO", "TEMPADC", "MAXRATIO", "MAXMPDADC"]

outputs = {}
for key, val in zip(["LT", "RT", "HT"], [0, 1, 2]):
    feats = np.column_stack([data[:, col_idx[col]] for col in feature_cols])
    feats = np.column_stack([feats, np.full((feats.shape[0],), val, dtype=np.float32)])

    X_scaled = scale_X(feats)

    ort_inputs = {"input": X_scaled}
    y_pred_scaled = ort_session.run(None, ort_inputs)[0]

    outputs[key] = inverse_scale_y(y_pred_scaled)

HighBias = outputs["HT"][:, 0]
HighRatio = outputs["HT"][:, 1]
HighTempAdc = outputs["HT"][:, 2]
NormalBias = outputs["RT"][:, 0]
NormalRatio = outputs["RT"][:, 1]
NormalTempAdc = outputs["RT"][:, 2]
LowBias = outputs["LT"][:, 0]
LowRatio = outputs["LT"][:, 1]
LowTempAdc = outputs["LT"][:, 2]

HighTempAdc[:] = int(np.mean(HighTempAdc))
NormalTempAdc[:] = int(np.mean(NormalTempAdc))
LowTempAdc[:] = int(np.mean(LowTempAdc))
HighRatio[2:] += 0.03
LowRatio[2:] += 0.03
NormalRatio += 0.03

pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
txlop = data[:, col_idx["MAXPOWER"]]

for i, j in pairs:
    avg_power = float(np.mean([txlop[i], txlop[j]]))
    if avg_power < 4.5:
        HighBias[i] = HighBias[j] = int(HighBias[i] + 150)
 
results = np.column_stack(
    [HighBias, HighRatio, HighTempAdc, NormalBias, NormalRatio, NormalTempAdc, LowBias, LowRatio, LowTempAdc])

bias_cols = [0, 3, 6]  # HighBias, NormalBias, LowBias
# 偏置和PD列做平均（按配对）
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]  # 按行配对
for col_idx in bias_cols:  # 遍历偏置列
    for i, j in pairs:
        avg = int(np.mean([results[i, col_idx], results[j, col_idx]]))
        results[i, col_idx] = results[j, col_idx] = avg

# ========= 保存结果 =========
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

np.savetxt(output_file,
           results,
           fmt="%d\t%.6f\t%d\t%d\t%.6f\t%d\t%d\t%.6f\t%d",
           header="HighBias\tHighRatio\tHighTempAdc\tNormalBias\tNormalRatio\tNormalTempAdc\tLowBias\tLowRatio\tLowTempAdc",
           comments="",
           encoding="utf-8")

print("✅ 已保存：", output_file)