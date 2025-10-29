# train_v2.py
import os
import json
import joblib
import math
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

pl.seed_everything(42)

"""
v2 说明:
- 集成 FiBiNet + FeatureGate
- 动态自蒸馏 (alpha_schedule: cosine)
- 蒸馏损失采用 MSE(student, teacher)
- AdamW + CosineAnnealingWarmRestarts scheduler
- 优化 Optuna objective 的 DataLoader 与返回值
- ONNX 导出前移动到 CPU
"""

# -------------------------
# 模型基础模块
# -------------------------
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x_in = x.unsqueeze(1)
        attn_output, _ = self.mha(x_in, x_in, x_in)
        out = self.norm(x_in + self.dropout(attn_output))
        return out.squeeze(1)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 2
        self.attn = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.attn(x)
        out = self.ffn(x)
        return self.norm(x + out)


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.cross_weights = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.cross_bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.cross_weights, self.cross_bias):
            xw = w(x)
            x = x0 * xw + b + x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, norm_type="layer"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(1, out_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.match_dims = (in_dim == out_dim)
        if not self.match_dims:
            self.res_connection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        # BatchNorm1d supports (N, C) and LayerNorm supports (N, C)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return x + out if self.match_dims else self.res_connection(x) + out


# -------------------------
# FiBiNet & Feature Gate
# -------------------------
class SENetGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (batch, dim)
        # 使用 batch-wise pooling -> learnable gate
        s = x.mean(dim=0, keepdim=True)  # (1, dim)
        s = F.relu(self.fc1(s))
        g = torch.sigmoid(self.fc2(s))  # (1, dim)
        return x * g  # 广播

class FiBiNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, input_dim // 2)
        self.senet = SENetGate(input_dim, reduction=4)
        self.linear_v = nn.Linear(input_dim, hidden_dim)
        self.linear_b = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x_g = self.senet(x)
        v = F.relu(self.linear_v(x_g))
        b = F.relu(self.linear_b(x_g))
        inter = v * b
        out = self.fc_out(inter)
        return x + out

class FeatureGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 初始化小权重
        self.gate = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        gates = torch.sigmoid(self.gate)  # (dim,)
        return x * gates
    
# -------------------------
# Lightning 模型 (增强版)
# -------------------------
class MLPLightning(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rates,
                 lr=1e-4, l2_reg=1e-4, huber_ratio=0.3, mse_ratio=0.7,
                 use_adv=False, adv_mode="FGM", adv_eps=1e-2, adv_ratio=0.5,
                 adv_steps=3, adv_alpha=1e-3,
                 feature_gate=False, use_fibinet=False,
                 distill_beta=0.1, teacher_model=None, alpha_schedule=None):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher_model", "alpha_schedule"])
        self.lr = lr
        self.l2_reg = l2_reg
        self.huber_ratio = huber_ratio
        self.mse_ratio = mse_ratio
        self.use_adv = use_adv
        self.adv_mode = adv_mode
        self.adv_eps = adv_eps
        self.adv_ratio = adv_ratio
        self.adv_steps = adv_steps
        self.adv_alpha = adv_alpha

        self.feature_gate_enabled = feature_gate
        self.use_fibinet = use_fibinet

        self.distill_beta = distill_beta
        self.teacher_model = teacher_model  # external teacher, set to eval() outside
        self.alpha_schedule = alpha_schedule

        # build model
        self.model = self._build_model(input_dim, output_dim, hidden_dims, dropout_rates)

        self.loss_huber = nn.HuberLoss()
        self.loss_mse = nn.MSELoss()

    def _build_model(self, input_dim, output_dim, hidden_dims, dropout_rates):
        layers = []
        if self.feature_gate_enabled:
            layers.append(FeatureGate(input_dim))
        # FiBiNet 在 CrossNet 之前
        if self.use_fibinet:
            layers.append(FiBiNetBlock(input_dim))

        layers.append(CrossNet(input_dim))
        last_dim = input_dim

        for i, (h_dim, dr) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(ResidualBlock(last_dim, h_dim, dropout=dr))
            layers.append(ResidualBlock(h_dim, h_dim, dropout=dr))
            last_dim = h_dim

            if i % 2 == 0:
                layers.append(TransformerEncoderBlock(last_dim, num_heads=2, ffn_dim=last_dim * 2, dropout=dr))

        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        # CosineAnnealingWarmRestarts with reasonable defaults
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=self.lr * 1e-4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _compute_main_loss(self, y_hat, y):
        return (self.huber_ratio * self.loss_huber(y_hat, y) +
                self.mse_ratio * self.loss_mse(y_hat, y))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss_main = self._compute_main_loss(y_hat, y)

        # 蒸馏损失
        loss_distill = 0.0
        if (self.teacher_model is not None) and self.teacher_model is not False:
            with torch.no_grad():
                teacher_pred = self.teacher_model(x)
            loss_distill = nn.MSELoss()(y_hat, teacher_pred)

        # 动态 alpha
        alpha = 1.0
        if self.alpha_schedule is not None:
            alpha = float(self.alpha_schedule(self.current_epoch if hasattr(self, "current_epoch") else 0))

        # ✅ FeatureGate L1 惩罚
        lambda_gate = 0.7  # 惩罚系数，可调
        gate_penalty = 0.0
        if self.feature_gate_enabled:
            for module in self.model:
                if isinstance(module, FeatureGate):
                    #print(module.gate.data.cpu().numpy())
                    gate_penalty += torch.sum(torch.abs(module.gate))

        loss = alpha * loss_main + self.distill_beta * loss_distill + lambda_gate * gate_penalty

        # 对抗训练保持不变
        if self.use_adv:
            x_adv = x.detach().clone().requires_grad_(True)
            if self.adv_mode == "FGM":
                adv_loss = self._compute_main_loss(self(x_adv), y)
                grad = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
                x_adv = (x_adv + self.adv_eps * grad.sign()).detach()
            elif self.adv_mode == "PGD":
                x_adv = x_adv + 0.001 * torch.randn_like(x_adv)
                for _ in range(self.adv_steps):
                    x_adv.requires_grad_(True)
                    adv_loss = self._compute_main_loss(self(x_adv), y)
                    grad = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
                    x_adv = (x_adv + self.adv_alpha * grad.sign())
                    perturb = torch.clamp(x_adv - x, min=-self.adv_eps, max=self.adv_eps)
                    x_adv = (x + perturb).detach()
            y_hat_adv = self(x_adv)
            adv_loss_val = self._compute_main_loss(y_hat_adv, y)
            loss = (1 - self.adv_ratio) * loss + self.adv_ratio * adv_loss_val

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_main_loss(y_hat, y)
        mae = nn.L1Loss()(y_hat, y)
        ss_res = torch.sum((y_hat - y) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)
        self.log("val_r2", r2, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_mae": mae, "val_r2": r2}

# -------------------------
# 数据加载与预处理
# -------------------------
def load_config(cfgmode, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config[cfgmode]

def load_and_preprocess_data(file_path, identifier, config_path, cfgmode, test_size=0.2, val_size=0.2):
    config = load_config(cfgmode, config_path)
    df = pd.read_csv(file_path, sep="\t", engine="python")
    df.columns = df.columns.str.upper()

    for col in ["RT_PDADC", "LT_PDADC", "HT_PDADC"]:
        mp_col = col.replace("PDADC", "MPDADC")
        ratio_col = col.replace("PDADC", "RATIO")
        if col in df.columns and mp_col in df.columns:
            df[ratio_col] = (df[mp_col] / df[col].replace(0, np.nan)) * 1000
   
    custom_state_encoding = {"LT": 0, "RT": 1, "HT": 2}
    dfs = []

    for state, mapping in config["states"].items():
        df_state = df[config["features"]].copy()
        for old_col, new_col in mapping.items():
            if old_col in df.columns:
                df_state[new_col] = df[old_col]
        if state in custom_state_encoding:
            df_state["状态"] = custom_state_encoding[state]
        else:
            raise ValueError(f"未定义的状态: {state}")
        dfs.append(df_state)

    df_long = pd.concat(dfs, ignore_index=True)
    unique_states = sorted(df_long["状态"].unique())
    state_map = {old: new for new, old in enumerate(unique_states)}
    df_long["状态"] = df_long["状态"].map(state_map)

    final_features = config["features"] + ["状态"]
    X = df_long[final_features].copy()
    y = df_long[config["target"]].copy()
    nan_idx = np.where(pd.isna(y).values)[0]  # NaN 行索引
    print(f"y 总数: {len(y)}, NaN 总数: {len(nan_idx)}")
    # Ensure y is 2D
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, pd.DataFrame):
        y = y.values
    else:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

    X = X.values  # numpy array
    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    print("最终特征列：", final_features)
    print("X_train.shape =", X_train.shape)
    print("scaler_X.mean_.shape =", scaler_X.mean_.shape)
    print("y_train.shape =", y_train.shape)
    print("检查训练数据 NaN / Inf：")
    print("X_train NaN:", np.isnan(X_train).sum(), "Inf:", np.isinf(X_train).sum())
    print("y_train NaN:", np.isnan(y_train).sum(), "Inf:", np.isinf(y_train).sum())
    print("X_val NaN:", np.isnan(X_val).sum(), "Inf:", np.isinf(X_val).sum())
    print("y_val NaN:", np.isnan(y_val).sum(), "Inf:", np.isinf(y_val).sum())
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

# -------------------------
# Optuna objective
# -------------------------
def objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim, device):
    # hyperparams
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [trial.suggest_int(f"n_units_l{i}", 64, 512, step=64) for i in range(n_layers)]
    dropout_rates = [trial.suggest_float(f"dropout_l{i}", 0.1, 0.4) for i in range(n_layers)]
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)

    huber_ratio = trial.suggest_float("huber_ratio", 0.1, 0.9)
    mse_ratio = 1.0 - huber_ratio

    use_adv = trial.suggest_categorical("use_adv", [True, False])
    if use_adv:
        adv_mode = trial.suggest_categorical("adv_mode", ["FGM", "PGD"])
        adv_eps = trial.suggest_float("adv_eps", 1e-3, 5e-2, log=True)
        adv_ratio = trial.suggest_float("adv_ratio", 0.2, 0.7)
        adv_steps = trial.suggest_int("adv_steps", 1, 5) if adv_mode == "PGD" else 0
        adv_alpha = trial.suggest_float("adv_alpha", 1e-4, 1e-2, log=True) if adv_mode == "PGD" else 0.0
    else:
        adv_mode, adv_eps, adv_ratio, adv_steps, adv_alpha = "FGM", 0.0, 0.0, 0, 0.0

    # build model
    model = MLPLightning(
        input_dim, output_dim, hidden_dims, dropout_rates,
        lr=lr, l2_reg=l2_reg, huber_ratio=huber_ratio, mse_ratio=mse_ratio,
        use_adv=use_adv, adv_mode=adv_mode, adv_eps=adv_eps, adv_ratio=adv_ratio,
        adv_steps=adv_steps, adv_alpha=adv_alpha,
        feature_gate=False, use_fibinet=False,
        distill_beta=0.0, teacher_model=None, alpha_schedule=None
    )

    # DataLoaders: ensure CPU tensors to avoid device issues
    if isinstance(X_train, np.ndarray):
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
    else:
        # already tensors
        X_train_t, y_train_t, X_val_t, y_val_t = X_train.cpu(), y_train.cpu(), X_val.cpu(), y_val.cpu()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=150, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=150)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    trainer = pl.Trainer(
        max_epochs=20,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_checkpointing=False,
        logger=False,
        callbacks=[early_stop],
        enable_model_summary=False,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        # 如果没有 val_loss，抛弃该 trial
        raise optuna.TrialPruned()
    return float(val_loss)


def run_optuna(X_train, y_train, X_val, y_val, input_dim, output_dim, device, n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim, device),
                   n_trials=n_trials)
    best_trial = study.best_trial
    rank_zero_info("\n🔹 最佳超参数:")
    for k, v in best_trial.params.items():
        rank_zero_info(f"  {k}: {v}")
    return best_trial


# -------------------------
# 自蒸馏主流程
# -------------------------
def cosine_alpha_schedule(epoch, max_epochs=100, alpha_init=0.3, alpha_final=0.3):
    # epoch in [0, max_epochs]
    t = min(epoch / max_epochs, 1.0)
    return float(alpha_final + 0.5 * (alpha_init - alpha_final) * (1 + math.cos(math.pi * t)))


def train_with_self_distillation(X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np,
                                 scaler_X, scaler_y, model_dir, identifier,
                                 distill_alpha_init=0.3, distill_alpha_final=0.3,
                                 distill_beta=0.15, n_rounds=2, n_trials=10, max_epochs=100):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep CPU numpy until DataLoader time
    input_dim, output_dim = X_train_np.shape[1], y_train_np.shape[1]
    current_y_train = y_train_np.copy()  # still numpy (标准化后的)
    history = {"round": [], "MAE": [], "R2": [], "Huber_loss": [], "MSE_loss": []}

    # --------------------
    # 第一次调参（不使用蒸馏损失）
    # --------------------
    rank_zero_info("\n🔍 进行超参数搜索以确定最佳超参（第一次搜索）...")
    best_trial = run_optuna(X_train_np, current_y_train, X_val_np, y_val_np, input_dim, output_dim, device, n_trials=n_trials)

    # Fix some params
    huber_ratio = best_trial.params.get("huber_ratio", 0.3)
    mse_ratio = 1.0 - huber_ratio
    rank_zero_info(f"固定混合比例: huber_ratio={huber_ratio:.3f}, mse_ratio={mse_ratio:.3f}")

    # 蒸馏轮
    teacher = None
    for round_idx in range(n_rounds):
        rank_zero_info(f"\n🔁 蒸馏轮次 {round_idx + 1}/{n_rounds}")

        # reconstruct hyperparams from best_trial
        n_layers = best_trial.params["n_layers"]
        hidden_dims = [best_trial.params[f"n_units_l{i}"] for i in range(n_layers)]
        dropout_rates = [best_trial.params[f"dropout_l{i}"] for i in range(n_layers)]

        # callbacks
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=model_dir,
                                     filename=f"{identifier}_round{round_idx+1}" + "-{epoch:02d}-{val_loss:.4f}")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=max_epochs,
            enable_checkpointing=True,
            logger=False,
            callbacks=[early_stop, checkpoint],
            enable_model_summary=False,
            deterministic=True,
            precision=16 if torch.cuda.is_available() else 32  # 可选混合精度
        )

        # build student model and inject teacher (teacher should be torch.nn.Module or None)
        # teacher must be on same device and eval()
        teacher_model_for_student = None
        if teacher is not None:
            teacher_model_for_student = teacher.to(device)
            teacher_model_for_student.eval()
            # ensure teacher doesn't require grad
            for p in teacher_model_for_student.parameters():
                p.requires_grad = False

        # dynamic alpha schedule closure
        alpha_sched_fn = lambda epoch: cosine_alpha_schedule(epoch, max_epochs=max_epochs,
                                                             alpha_init=distill_alpha_init,
                                                             alpha_final=distill_alpha_final)

        model = MLPLightning(
            input_dim, output_dim, hidden_dims, dropout_rates,
            lr=best_trial.params.get("lr", 1e-4),
            l2_reg=best_trial.params.get("l2_reg", 1e-4),
            huber_ratio=huber_ratio,
            mse_ratio=mse_ratio,
            use_adv=best_trial.params.get("use_adv", False),
            adv_mode=best_trial.params.get("adv_mode", "FGM"),
            adv_eps=best_trial.params.get("adv_eps", 0.0),
            adv_ratio=best_trial.params.get("adv_ratio", 0.0),
            adv_steps=best_trial.params.get("adv_steps", 0),
            adv_alpha=best_trial.params.get("adv_alpha", 0.0),
            feature_gate=True,
            use_fibinet=True,
            distill_beta=distill_beta,
            teacher_model=teacher_model_for_student,
            alpha_schedule=alpha_sched_fn
        ).to(device)

        # DataLoaders: convert numpy to CPU tensors
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_t = torch.tensor(current_y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_np, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True, num_workers=0, persistent_workers=False)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=128, num_workers=0)

        trainer.fit(model, train_loader, val_loader)

        # load best model if checkpoint exists
        best_path = checkpoint.best_model_path
        if best_path and os.path.exists(best_path):
            model = MLPLightning.load_from_checkpoint(
                best_path,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout_rates=dropout_rates,
                feature_gate=True,
                use_fibinet=True,
                distill_beta=distill_beta,
                teacher_model=None,
                alpha_schedule=None,
                strict=False,  # 可选，但安全
            )
            model.to(device)
        else:
            rank_zero_info("没有找到 checkpoint，使用训练结束时的模型")


        # 生成 soft labels（teacher for next round）
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
            y_soft = model(X_train_tensor).cpu().numpy()  # still standardized
        # 更新训练目标：alpha_schedule 用 epoch=0 (we keep global α for combination)
        # 这里采用 distill_alpha_init/ final 作为 round-level combination via cosine at epoch=0 (or choose other policy)
        # 更直接：使用 distill_alpha_init as当前round的alpha起点
        alpha_round = cosine_alpha_schedule(0, max_epochs=1, alpha_init=distill_alpha_init, alpha_final=distill_alpha_final)
        # 使用 distill_alpha_init 作为保守策略
        alpha_round = distill_alpha_init if round_idx == 0 else (distill_alpha_final if round_idx == n_rounds - 1 else (distill_alpha_init + distill_alpha_final) / 2)
        # 更稳健：使用外部参数 distill_alpha_init, distill_alpha_final；这里保持 simple policy
        current_y_train = alpha_round * y_train_np + (1 - alpha_round) * y_soft

        # 评估验证集
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
            y_pred = model(X_val_tensor).cpu().numpy()
            huber_loss_val = nn.HuberLoss()(torch.tensor(y_pred), torch.tensor(y_val_np)).item()
            mse_loss_val = nn.MSELoss()(torch.tensor(y_pred), torch.tensor(y_val_np)).item()

        y_val_inv = scaler_y.inverse_transform(y_val_np)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        mae = mean_absolute_error(y_val_inv, y_pred_inv)
        r2 = r2_score(y_val_inv, y_pred_inv)
        rank_zero_info(f"📈 蒸馏轮 {round_idx + 1}: MAE={mae:.4f}, R2={r2:.4f}, Huber={huber_loss_val:.4f}, MSE={mse_loss_val:.4f}")

        history["round"].append(round_idx + 1)
        history["MAE"].append(mae)
        history["R2"].append(r2)
        history["Huber_loss"].append(huber_loss_val)
        history["MSE_loss"].append(mse_loss_val)

        # 保存 teacher（深拷贝或 load best model）
        if best_path and os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]

            # ✅ 过滤掉 teacher_model 相关的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("teacher_model.")}

            checkpoint["state_dict"] = filtered_state_dict

            # ✅ 加载时传入一致参数，并关闭 strict
            teacher = MLPLightning.load_from_checkpoint(
                checkpoint_path=best_path,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout_rates=dropout_rates,
                feature_gate=True,
                use_fibinet=True,
                distill_beta=distill_beta,
                teacher_model=None,          # 关键：防止递归嵌套
                alpha_schedule=None,
                strict=False
            )

            teacher.to(device)
        else:
            teacher = model
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # --------------------
    # 保存模型与 scaler
    # --------------------
    final_model = model
    torch.save(final_model.state_dict(), os.path.join(model_dir, "mlp_model.pt"))
    with open(os.path.join(model_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_trial.params, f, indent=4, ensure_ascii=False)
    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

    # --------------------
    # 测试集评估
    # --------------------
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_test_pred = final_model(X_test_tensor).cpu().numpy()

    y_test_inv = scaler_y.inverse_transform(y_test_np)
    y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)
    test_mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
    test_r2 = r2_score(y_test_inv, y_test_pred_inv)
    rank_zero_info(f"\n📝 测试集评估: MAE={test_mae:.4f}, R²={test_r2:.4f}")

    # --------------------
    # ONNX 导出（先移动到 CPU）
    # --------------------
    try:
        example_input = torch.tensor(X_test_np[:1], dtype=torch.float32)
        final_model_cpu = final_model.to("cpu").eval()
        onnx_path = os.path.join(model_dir, "mlp_model.onnx")
        torch.onnx.export(final_model_cpu, example_input, onnx_path,
                          export_params=True, opset_version=17, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        rank_zero_info("ONNX 导出成功: " + onnx_path)
    except Exception as e:
        rank_zero_info("ONNX 导出失败: " + str(e))

    return history

# -------------------------
# 主入口
# -------------------------
if __name__ == "__main__":
    file_path = r"D:\\AI_Prediction\\AiCode\\Data\\562-0435.txt"
    config_path = r"D:\\AI_Prediction\\AiCode\\PyTorch\\ModelsConfig.json"
    identifier = "562-0435"
    cfgmode = "AI_EML_2T"

    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = \
        load_and_preprocess_data(file_path, identifier, config_path, cfgmode)

    train_with_self_distillation(
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler_X, scaler_y,
        model_dir=f"./output/{identifier}",
        identifier=identifier,
        distill_alpha_init=0.9,
        distill_alpha_final=0.3,
        distill_beta=0.1,
        n_rounds=2,
        n_trials=10,
        max_epochs=100
    )