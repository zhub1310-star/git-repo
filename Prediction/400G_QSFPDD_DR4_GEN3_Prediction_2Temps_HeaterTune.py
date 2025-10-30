import os
import numpy as np
import onnxruntime as ort
import configparser

# 创建两个 ConfigParser 实例
config_login = configparser.ConfigParser()
config_model = configparser.ConfigParser()

# 读取 Login.ini 文件
login_file = r"D:\AtsTempoData\Login.ini"
config_login.read(login_file, encoding="utf-8")

# 获取 ModelPN（确保键存在）
if config_login.has_option("Login", "500PN"):
    ModelPN = config_login.get("Login", "500PN")
else:
    raise KeyError(f"Key '500PN' not found in section 'Login' of {login_file}")

# 读取 ModelConfig.ini 文件
model_file = r"D:\API\400G_QSFPDD_DR4_GEN3_Prediction_2Temps\ModelConfig.ini"
config_model.read(model_file, encoding="utf-8")

# 获取 ModelType（确保键存在）
if config_model.has_option("Model", ModelPN):
    ModelType = config_model.get("Model", ModelPN)
else:
    ModelType = "LP"

# ========= 加载模型和scaler =========
if ModelType == "Sumitomo":
    model_dir = r"D:\\API\\400G_QSFPDD_DR4_GEN3_Prediction_2Temps\\Sumitomo"
    # ========= 硬编码 scaler 参数 =========
    scaler_X_mean = np.array(
        [265.9225266774114, 1043.3236468033056, 2112.3121155513604, 0.49388468524305107, 1086.8140984255697,
         0.9121148239805801, 1926.9465175735352, 0.9994232540522859], dtype=np.float32)
    scaler_X_scale = np.array(
        [40.29681210839095, 62.769997274331125, 72.30935034369216, 0.02371600387008254, 36.45276223094661,
         0.01854108560575196, 80.35862150670735, 0.8165016486077978], dtype=np.float32)

    scaler_y_mean = np.array([1637.7951346744828, 0.4947266293166954, 1084.1417331646137], dtype=np.float32)
    scaler_y_scale = np.array([413.87403455271766, 0.051994855454112164, 108.65342143205021], dtype=np.float32)

# ========= 加载模型和scaler =========
if ModelType == "SumitomoD411":
    model_dir = r"D:\\API\\400G_QSFPDD_DR4_GEN3_Prediction_2Temps\\SumitomoD411"
    # ========= 硬编码 scaler 参数 =========
    scaler_X_mean = np.array(
        [305.531324676859, 716.2724099699286, 699.2757777027634, 1.0266676346037509, 1089.5272595281306, 
         1.909518313200242, 1332.8216091954023, 1.0036297640653358], dtype=np.float32)
    scaler_X_scale = np.array(
        [37.94922878585404, 65.27643302038007, 65.33858341367846, 0.06713286479559188, 36.4006862390328, 
         0.0924048301258583, 111.40169386648284, 0.8175251564182311], dtype=np.float32)

    scaler_y_mean = np.array([1964.2425408348458, 0.7603549912465276, 1085.1808832425893], dtype=np.float32)
    scaler_y_scale = np.array([467.4738105319806, 0.05467640738744527, 109.2779012199889], dtype=np.float32)

if ModelType == "Brcm":
    model_dir = r"D:\\API\\400G_QSFPDD_DR4_GEN3_Prediction_2Temps\\Brcm"
    # ========= 硬编码 scaler 参数 =========
    scaler_X_mean = np.array(
        [257.98323129309927, 1073.914374569047, 2165.8693258862777, 0.4957488833746898, 1090.3604962779157,
         0.9148626295533498, 1981.9933250620347, 0.9979156327543425], dtype=np.float32)
    scaler_X_scale = np.array(
        [36.65250721194042, 64.25343438886901, 68.08407153500058, 0.023696540302887027, 36.31047969572775,
         0.017204161547685882, 80.50952245582849, 0.8164533983584739], dtype=np.float32)

    scaler_y_mean = np.array([1576.8655334987593, 0.5023853173345015, 1087.4809677419355], dtype=np.float32)
    scaler_y_scale = np.array([421.92242864029856, 0.04489116112939955, 108.43706644735757], dtype=np.float32)

if ModelType == "BrcmD411":
    model_dir = r"D:\\API\\400G_QSFPDD_DR4_GEN3_Prediction_2Temps\\BrcmD411"
    # ========= 硬编码 scaler 参数 =========
    scaler_X_mean = np.array(
        [187.02641765215063, 1114.942515389204, 1457.6944852225763, 0.7662264855529275, 1090.0063949960322, 
         1.4213520215656137, 2067.754866265229, 1.0012136488820427], dtype=np.float32)
    scaler_X_scale = np.array(
        [19.07427483781621, 67.61059169459082, 87.77891907079889, 0.045256301279427896, 36.090486028988735, 
         0.06439261637785376, 86.33081881777743, 0.8169529084187953], dtype=np.float32)

    scaler_y_mean = np.array([1266.4491434439622, 0.7767509370089313, 1085.3275451617421], dtype=np.float32)
    scaler_y_scale = np.array([409.11573275394875, 0.06560143903653184, 109.99204598716092], dtype=np.float32)

out_predict_path = r"D:\\AtsTempoData"
onnx_path = os.path.join(model_dir, "mlp_model.onnx")
#params_path = os.path.join(model_dir, "params.json")

def scale_X(X):
    return (X - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(y_scaled):
    return y_scaled * scaler_y_scale + scaler_y_mean

# ========= 创建 ONNX 推理会话 =========
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# ========= 加载数据 =========
pred_file_path = r"D:\\AtsTempoData\\PredictedData.txt"

# 先读取第一行获取列名
with open(pred_file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    header = [h.upper() for h in header]

# 把数据读进 numpy
data = np.loadtxt(pred_file_path, delimiter="\t", skiprows=1, dtype=np.float32)

# 找到所需列索引
col_idx = {name: i for i, name in enumerate(header)}
bias_pwr = data[:, col_idx["BIASDAC"]] / data[:, col_idx["MAXPOWER"]]
data = np.column_stack([data, bias_pwr])
header.append("BIASPWR")
col_idx["BIASPWR"] = len(header) - 1

feature_cols = ["BIASPWR", "REALMPDADC", "REALPDADC", "REALRATIO", "TEMPADC", "MAXRATIO", "MAXMPDADC"]

# 预测
# ========= 构造 LT、HT 数据 =========
outputs = {}
for key, val in zip(["LT", "RT", "HT"], [0, 1, 2]):  # 三种温度
    feats = np.column_stack([data[:, col_idx[col]] for col in feature_cols])
    feats = np.column_stack([feats, np.full((feats.shape[0],), val, dtype=np.float32)])

    # 标准化
    X_scaled = scale_X(feats)

    # 推理
    ort_inputs = {"input": X_scaled}
    y_pred_scaled = ort_session.run(None, ort_inputs)[0]

    # 反标准化
    outputs[key] = inverse_scale_y(y_pred_scaled)

# 构建结果表
HighBias = outputs["HT"][:, 0]
HighRatio = outputs["HT"][:, 1]
HighTempAdc = outputs["HT"][:, 2]
NormalBias = outputs["RT"][:, 0]
NormalRatio = outputs["RT"][:, 1]
NormalTempAdc = outputs["RT"][:, 2]
LowBias = outputs["LT"][:, 0]
LowRatio = outputs["LT"][:, 1]
LowTempAdc = outputs["LT"][:, 2]

# 平均温度ADC为整数
HighTempAdc[:] = int(np.mean(HighTempAdc))
NormalTempAdc[:] = int(np.mean(NormalTempAdc))
LowTempAdc[:] = int(np.mean(LowTempAdc))

if ModelType == "SumitomoD411":
    HighBias[:] = int(np.mean(HighBias))
    NormalBias[:] = int(np.mean(NormalBias))
    LowBias[:] = int(np.mean(LowBias))
    LowRatio[0] += 0.03
    LowRatio[1] += 0.03
    # ========= 判断条件: MAXPOWER < 4 时，LowBias + 140 =========
    if "MAXPOWER" in col_idx:
        txlop = data[:, col_idx["MAXPOWER"]]  # 取出每行的 TXLOP 值
        
        # 如果任意一个值 < 4，则只加一次 140
        if any(txlop[i] < 4 for i in range(min(len(txlop), 4))):
            LowBias += 140
            NormalBias += 140
            LowRatio[0] += 0.02
            LowRatio[1] += 0.02
        # 如果任意一个值 < 4，则执行 HighBias 操作（这里只是示例）
        if any(txlop[i] < 4 for i in range(min(len(txlop), 4))):
            HighBias += 200
            HighBias = np.minimum(HighBias, 2886)
            HighRatio[0] += 0.02
            HighRatio[1] += 0.02

if ModelType == "BrcmD411v4":
    HighBias[:] = int(np.mean(HighBias))
    NormalBias[:] = int(np.mean(NormalBias))
    LowBias[:] = int(np.mean(LowBias))
    LowRatio += 0.04
    
# Sumitomo 特殊修正
if ModelType == "Sumitomo":
    LowRatio += 0.04
    LowRatio[0] += 0.01
# Brcm 特殊修正
if ModelType == "Brcm":
    LowRatio += 0.04
# 拼成二维数组
results = np.column_stack(
    [HighBias, HighRatio, HighTempAdc, NormalBias, NormalRatio, NormalTempAdc, LowBias, LowRatio, LowTempAdc])

bias_cols = [0, 3, 6]  # HighBias, NormalBias, LowBias
# 偏置和PD列做平均（按配对）

if ModelType != "SumitomoD411" and ModelType != "BrcmD411v4":
    pairs = [(0, 1), (2, 3)]  # 按行配对
    for col_idx in bias_cols:  # 遍历偏置列
        for i, j in pairs:
            avg = int(np.mean([results[i, col_idx], results[j, col_idx]]))
            results[i, col_idx] = results[j, col_idx] = avg

# ========= 保存结果 =========
output_file = os.path.join(out_predict_path, "Predictions.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

np.savetxt(output_file,
           results,
           fmt="%.6f\t%.6f\t%d\t%.6f\t%.6f\t%d\t%.6f\t%.6f\t%d",
           header="HighBias\tHighRatio\tHighTempAdc\tNormalBias\tNormalRatio\tNormalTempAdc\tLowBias\tLowRatio\tLowTempAdc",
           comments="",
           encoding="utf-8")

print("✅ 已保存：", output_file)
