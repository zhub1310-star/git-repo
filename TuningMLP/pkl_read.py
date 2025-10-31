import joblib
import numpy as np

scaler_X = joblib.load(r"D:/AI_Prediction/AiCode/ModelStorage/562-0431A/scaler_X.pkl")
scaler_y = joblib.load(r"D:/AI_Prediction/AiCode/ModelStorage/562-0431A/scaler_y.pkl")

print("scaler_X_mean =", scaler_X.mean_.tolist())
print("scaler_X_scale =", scaler_X.scale_.tolist())
print("scaler_y_mean =", scaler_y.mean_.tolist())
print("scaler_y_scale =", scaler_y.scale_.tolist())


print("\nScaler_X mean shape:", scaler_X.mean_.shape)
print("Scaler_X scale shape:", scaler_X.scale_.shape)
print("Scaler_y mean shape:", scaler_y.mean_.shape)
print("Scaler_y scale shape:", scaler_y.scale_.shape)
