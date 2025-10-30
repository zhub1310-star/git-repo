import os
import sys
import numpy as np
import onnxruntime as ort
import configparser

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

config_login = configparser.ConfigParser()
config_model = configparser.ConfigParser()

login_file = r"D:\AtsTempoData\Login.ini"
config_login.read(login_file, encoding="utf-8")

if config_login.has_option("Login", "500PN"):
    ModelPN = config_login.get("Login", "500PN")
else:
    raise KeyError(f"Key '500PN' not found in section 'Login' of {login_file}")

model_file = os.path.join(base_dir, "ModelConfig.ini")
config_model.read(model_file, encoding="utf-8")

if config_model.has_option("Model", ModelPN):
    ModelType = config_model.get("Model", ModelPN)
else:
    ModelType = "LP"

if ModelType == "Landmark":
    model_dir = os.path.join(base_dir, "Landmark.onnx")

    scaler_X_mean = np.array(
        [215.4482486774288, 619.3823114633938, 1483.9291505488984, 0.41729836442794205, 1057.0016473387634, 
         0.9193926048217345, 1364.4790225790005, 0.4973917136245375], dtype=np.float32)
    scaler_X_scale = np.array(
        [28.972534249108, 46.770385695105645, 62.13019623489838, 0.024708938589760615, 36.75740362572873, 
         0.01817685586405228, 67.80739312700568, 0.4999931967959], dtype=np.float32)

    scaler_y_mean = np.array([175.12960636764078, 0.30488039794872346, 1132.3302391256032], dtype=np.float32)
    scaler_y_scale = np.array([58.727061856485435, 0.015668285985054747, 83.73167792250744], dtype=np.float32)
else:
    model_dir = os.path.join(base_dir, "Brcm.onnx")

    scaler_X_mean = np.array(
        [243.3228869827127, 446.02919738080095, 1385.2392868912937, 0.3219397949915661, 1057.115998442974, 
         0.9075676792526275, 1257.3294840188573, 0.5000216253622248], dtype=np.float32)
    scaler_X_scale = np.array(
        [33.82572650423334, 37.319538899835564, 59.44593789641309, 0.02237904761562378, 35.90467588320394, 
         0.020083941568467663, 62.35211507655099, 0.4999999995323437], dtype=np.float32)

    scaler_y_mean = np.array([261.11037090345576, 0.3017271826324339, 1126.2287098308896], dtype=np.float32)
    scaler_y_scale = np.array([78.99080003664278, 0.017106719306638263, 77.39510832886148], dtype=np.float32)

out_predict_path = r"D:\\AtsTempoData"
onnx_path = model_dir

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

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

feature_cols = ["BIASPWR", "REALMPDADC", "REALPDADC", "REALRATIO", "TEMPADC", "MAXRATIO", "MAXMPDADC"]

outputs = {}
for key, val in zip(["RT", "HT"], [0, 1]):
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

HighTempAdc[:] = int(np.mean(HighTempAdc))
NormalTempAdc[:] = int(np.mean(NormalTempAdc))
MaxPower = data[:, col_idx["MAXPOWER"]]
HighBias = HighBias * MaxPower
NormalBias = NormalBias * MaxPower

# Sumitomo 特殊修正
if ModelType != "Landmark":
    HighRatio += 0.03

results = np.column_stack(
    [HighBias, HighRatio, HighTempAdc, NormalBias, NormalRatio, NormalTempAdc])

bias_cols = [0, 3]  
pairs = [(0, 1), (2, 3)]  
for col_idx in bias_cols:
    for i, j in pairs:
        avg = int(np.mean([results[i, col_idx], results[j, col_idx]]))
        results[i, col_idx] = results[j, col_idx] = avg

output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

np.savetxt(output_file,
           results,
           fmt="%.6f\t%.6f\t%d\t%.6f\t%.6f\t%d",
           header="HighBias\tHighRatio\tHighTempAdc\tNormalBias\tNormalRatio\tNormalTempAdc",
           comments="",
           encoding="utf-8")

print("✅ 已保存：", output_file)
