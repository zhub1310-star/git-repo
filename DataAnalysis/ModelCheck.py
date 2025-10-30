import sys
import joblib
import json
import numpy as np
from pathlib import Path
import pandas as pd
import onnxruntime as ort
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QSpinBox, QMessageBox, QTabWidget, QFrame, QComboBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")


class RegressionMonitorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型监控与评估工具")
        self.resize(1350, 750)

        # ---------------- 数据 & 模型 ----------------
        self.data = None
        self.scaler_X = None
        self.scaler_y = None
        self.ort_session = None
        self.feature_cols = []
        self.target_col = None
        self.history_metrics = pd.DataFrame(columns=["MSE", "MAE", "R2"])
        self.last_selected_features = []

        # ---------------- 主布局 ----------------
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # ---------------- 左侧控制栏 ----------------
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("background-color: #2c3e50; color: white;")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # 数据上传
        self.upload_data_btn = QPushButton("数据上传")
        self.upload_data_btn.setStyleSheet("background-color: #3498db; color: white; padding:8px;")
        self.upload_data_btn.clicked.connect(self.load_data)
        left_layout.addWidget(self.upload_data_btn)

        # 分隔符选择
        left_layout.addWidget(QLabel("选择分隔符"))
        self.sep_combo = QComboBox()
        self.sep_combo.addItems(["空格", ",", "\t"])
        left_layout.addWidget(self.sep_combo)

        # 生成状态列
        self.generate_state_btn = QPushButton("生成状态列")
        self.generate_state_btn.setStyleSheet("background-color: #8e44ad; color: white; padding:8px;")
        self.generate_state_btn.clicked.connect(self.generate_state_column)
        left_layout.addWidget(self.generate_state_btn)

        # 上传模型文件夹
        self.upload_model_folder_btn = QPushButton("选择模型文件夹")
        self.upload_model_folder_btn.setStyleSheet("background-color: #3498db; color: white; padding:8px;")
        self.upload_model_folder_btn.clicked.connect(self.load_model_folder)
        left_layout.addWidget(self.upload_model_folder_btn)

        # 上传配置文件
        self.upload_model_config_btn = QPushButton("选择配置文件")
        self.upload_model_config_btn.setStyleSheet("background-color: #3498db; color: white; padding:8px;")
        self.upload_model_config_btn.clicked.connect(self.load_model_config)
        left_layout.addWidget(self.upload_model_config_btn)

        # 配置选择下拉列表
        left_layout.addWidget(QLabel("选择配置模式"))
        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self.update_config_display)
        left_layout.addWidget(self.config_combo)

        # 特征列显示
        left_layout.addWidget(QLabel("特征列"))
        self.feature_display = QLabel()
        self.feature_display.setStyleSheet("background-color: #34495e; padding: 5px; border-radius: 5px;")
        self.feature_display.setWordWrap(True)
        left_layout.addWidget(self.feature_display)

        # 目标列显示
        left_layout.addWidget(QLabel("目标列"))
        self.target_display = QLabel()
        self.target_display.setStyleSheet("background-color: #34495e; padding: 5px; border-radius: 5px;")
        left_layout.addWidget(self.target_display)

        # 阈值设置
        left_layout.addWidget(QLabel("MSE 阈值"))
        self.mse_threshold = QSpinBox()
        self.mse_threshold.setMaximum(1000000)
        self.mse_threshold.setValue(100)
        left_layout.addWidget(self.mse_threshold)

        left_layout.addWidget(QLabel("R² 阈值"))
        self.r2_threshold = QLineEdit()
        self.r2_threshold.setText("0.8")
        left_layout.addWidget(self.r2_threshold)

        # 评估按钮
        self.evaluate_btn = QPushButton("评估模型")
        self.evaluate_btn.setStyleSheet("background-color: #e67e22; color: white; padding:8px;")
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        left_layout.addWidget(self.evaluate_btn)
        left_layout.addStretch()

        # ---------------- 右侧主区域 ----------------
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 2)

        # 标签页
        self.tabs = QTabWidget()
        right_panel.addWidget(self.tabs)

        # 指标页
        self.tab_metrics = QWidget()
        self.tabs.addTab(self.tab_metrics, "指标")
        tab_metrics_layout = QVBoxLayout()
        self.tab_metrics.setLayout(tab_metrics_layout)
        self.metrics_card = QLabel()
        self.metrics_card.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metrics_card.setFont(QFont("Arial", 12))
        self.metrics_card.setStyleSheet("background-color: #ecf0f1; border-radius: 10px; padding: 15px;")
        tab_metrics_layout.addWidget(self.metrics_card)

        # 图表页
        self.tab_plots = QWidget()
        self.tabs.addTab(self.tab_plots, "图表")
        tab_plots_layout = QVBoxLayout()
        self.tab_plots.setLayout(tab_plots_layout)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(9, 4))
        self.canvas = FigureCanvas(self.fig)
        tab_plots_layout.addWidget(self.canvas)

        # 历史监控页
        self.tab_history = QWidget()
        self.tabs.addTab(self.tab_history, "历史监控")
        tab_history_layout = QVBoxLayout()
        self.tab_history.setLayout(tab_history_layout)
        self.fig_history, self.ax_history = plt.subplots(figsize=(9, 3))
        self.canvas_history = FigureCanvas(self.fig_history)
        tab_history_layout.addWidget(self.canvas_history)

    # ---------------- 方法 ----------------
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择TXT文件", "", "TXT Files (*.txt)")
        if file_path:
            sep = self.sep_combo.currentText()
            if sep == "空格":
                sep = r"\s+"
            self.data = pd.read_csv(file_path, sep=sep, engine='python')
            self.data.columns = self.data.columns.str.upper()

        for col in ["RT_PDADC", "LT_PDADC", "HT_PDADC"]:
            mp_col = col.replace("PDADC", "MPDADC")
            ratio_col = col.replace("PDADC", "RATIO")
            if col in self.data.columns and mp_col in self.data.columns:
                self.data[ratio_col] = (self.data[mp_col] / self.data[col].replace(0, np.nan)) * 1000

        QMessageBox.information(self, "提示", "数据加载成功！请生成状态列（如需要）并选择特征。")

    def generate_state_column(self):
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先上传数据！")
            return

        cfg_name = self.config_combo.currentText()
        if not cfg_name or not hasattr(self, "all_configs"):
            QMessageBox.warning(self, "警告", "请先加载配置文件并选择配置模式！")
            return

        config = self.all_configs.get(cfg_name)
        if not config:
            QMessageBox.warning(self, "错误", f"未找到配置 '{cfg_name}'")
            return

        # 状态编码映射
        custom_state_encoding = {"LT": 0, "RT": 1, "HT": 2}
        dfs = []

        # 目标列
        target_cols = config["target"] if isinstance(config["target"], list) else [config["target"]]

        # 遍历每个状态配置
        for state_name, mapping in config.get("states", {}).items():
            # 从原始数据复制整份，后续生成映射列
            df_state = self.data.copy()

            # 根据映射关系生成新列
            for old_col, new_col in mapping.items():
                if old_col in self.data.columns:
                    df_state[new_col] = self.data[old_col]
                else:
                    print(f"⚠️ 警告：原始数据中缺少列 {old_col}")

            # 添加状态列
            if state_name in custom_state_encoding:
                df_state["状态"] = custom_state_encoding[state_name]
            else:
                df_state["状态"] = state_name

            dfs.append(df_state)

        # 拼接所有状态数据
        df_long = pd.concat(dfs, ignore_index=True)

        # 编码状态列（数值化）
        unique_states = sorted(df_long["状态"].unique())
        state_map = {old: new for new, old in enumerate(unique_states)}
        df_long["状态"] = df_long["状态"].map(state_map)

        # 最终特征列（与训练一致）
        self.feature_cols_final = config["features"] + ["状态"]
        self.feature_cols = config["features"]
        self.target_col = config["target"]

        # 提取特征和目标列（缺失列自动跳过）
        missing_features = [c for c in self.feature_cols_final if c not in df_long.columns]
        if missing_features:
            QMessageBox.warning(
                self,
                "警告",
                f"部分特征列在数据中不存在: {', '.join(missing_features)}\n这些列将被跳过。"
            )
        available_features = [c for c in self.feature_cols_final if c in df_long.columns]
        available_target = [c for c in target_cols if c in df_long.columns]

        # 更新主数据
        self.data = df_long.copy()

        # 界面显示
        self.feature_display.setText(", ".join(available_features))
        self.target_display.setText(", ".join(available_target))

        # 保存中间文件以便排查
        #save_path = Path.cwd() / f"debug_state_data_{cfg_name}.csv"
        #self.data.to_csv(save_path, index=False, encoding="utf-8-sig")

        QMessageBox.information(
            self,
            "提示",
            f"状态列已生成，数据已拼接完成，共 {len(df_long)} 行。\n"
        )


    def load_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder:
            folder_path = Path(folder)
            onnx_file = folder_path / "mlp_model.onnx"
            scaler_X_file = folder_path / "scaler_X.pkl"
            scaler_y_file = folder_path / "scaler_y.pkl"
            if not (onnx_file.exists() and scaler_X_file.exists() and scaler_y_file.exists()):
                QMessageBox.warning(self, "错误", "文件夹中缺少 mlp_model.onnx 或 scaler_X.pkl / scaler_y.pkl")
                return
            self.ort_session = ort.InferenceSession(str(onnx_file))
            with open(scaler_X_file, "rb") as f:
                self.scaler_X = joblib.load(f)
            with open(scaler_y_file, "rb") as f:
                self.scaler_y = joblib.load(f)
            QMessageBox.information(self, "提示", "模型和标准化对象加载成功！")

    def load_model_config(self):
    
        folder = QFileDialog.getExistingDirectory(self, "选择配置文件")
        if folder:
            folder_path = Path(folder)
            config_path = folder_path / "ModelsConfig.json"

            # 读取 JSON
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.all_configs = json.load(f)  # 保存所有配置
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取配置文件失败: {e}")
                return

            # 清空下拉框并添加所有配置项
            self.config_combo.clear()
            for cfg_name in self.all_configs.keys():
                self.config_combo.addItem(cfg_name)

            QMessageBox.information(self, "提示", "配置文件加载成功，请从下拉列表选择配置！")

    def update_config_display(self):
        cfg_name = self.config_combo.currentText()
        if not cfg_name or not hasattr(self, "all_configs"):
            return

        config = self.all_configs.get(cfg_name, {})

        # 检查 target 是否存在
        if "target" not in config or "features" not in config:
            QMessageBox.warning(self, "错误", f"配置 '{cfg_name}' 中缺少 'target' 或 'features' 字段")
            return

        # 设置类属性
        self.feature_cols = config["features"]
        self.target_col = config["target"]

        # 最终特征列：加上状态列
        self.feature_cols_final = self.feature_cols + ["状态"]

        # 显示在界面上
        self.feature_display.setText(", ".join(self.feature_cols_final))
        if isinstance(self.target_col, list):
            self.target_display.setText(", ".join(self.target_col))
        else:
            self.target_display.setText(str(self.target_col))

    def select_all_features(self):
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(True)

    def deselect_all_features(self):
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(False)

    def evaluate_model(self):
        if self.data is None or self.scaler_X is None or self.scaler_y is None or self.ort_session is None:
            QMessageBox.warning(self, "警告", "请先上传数据和模型文件夹！")
            return

        print(f"当前数据行数: {len(self.data)}")

        # ===== 数据准备 =====
        X = self.data[self.feature_cols_final].values
        y = self.data[self.target_col].values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.ort_session.run(None, {"input": X_scaled.astype("float32")})[0]
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        target_names = self.target_col if isinstance(self.target_col, list) else [self.target_col]

        # ===== 阈值 =====
        mse_limit = self.mse_threshold.value()
        try:
            r2_limit = float(self.r2_threshold.text())
        except ValueError:
            r2_limit = 0.8

        # ===== 计算指标 =====
        metrics_text = []
        metrics_dict = {}
        for i, col_name in enumerate(target_names):
            mse = mean_squared_error(y[:, i], y_pred[:, i])
            mae = mean_absolute_error(y[:, i], y_pred[:, i])
            r2 = r2_score(y[:, i], y_pred[:, i])
            metrics_text.append(f"{col_name} → MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            metrics_dict[col_name] = {"MSE": mse, "MAE": mae, "R2": r2}

        self.metrics_card.setText("\n".join(metrics_text))

        # ===== 图表页 =====
        for i in reversed(range(self.tab_plots.layout().count())):
            widget = self.tab_plots.layout().itemAt(i).widget()
            if widget:
                widget.deleteLater()

        plot_tabs = QTabWidget()
        self.tab_plots.layout().addWidget(plot_tabs)

        for i, col_name in enumerate(target_names):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            fig, axs = plt.subplots(1, 2, figsize=(9, 4))
            canvas = FigureCanvas(fig)

            # ---- 左图：真实 vs 预测 ----
            axs[0].scatter(range(len(y[:, i])), y[:, i], label="真实值", alpha=0.6)
            axs[0].scatter(range(len(y_pred[:, i])), y_pred[:, i], label="预测值", alpha=0.6)
            axs[0].set_title(f"{col_name} - 真实 vs 预测")
            axs[0].legend()

            # ---- 右图：拟合曲线 ----
            axs[1].scatter(y[:, i], y_pred[:, i], alpha=0.6)
            axs[1].plot([y[:, i].min(), y[:, i].max()], [y[:, i].min(), y[:, i].max()], 'r--')
            axs[1].set_title(f"{col_name} - 拟合曲线")
            axs[1].set_xlabel("真实值")
            axs[1].set_ylabel("预测值")

            plt.tight_layout()
            layout.addWidget(canvas)
            plot_tabs.addTab(tab, col_name)

        # ===== 历史监控页 =====
        for i in reversed(range(self.tab_history.layout().count())):
            widget = self.tab_history.layout().itemAt(i).widget()
            if widget:
                widget.deleteLater()

        history_tabs = QTabWidget()
        self.tab_history.layout().addWidget(history_tabs)

        # 初始化或更新多目标历史记录
        if not hasattr(self, "history_metrics_multi"):
            self.history_metrics_multi = {name: pd.DataFrame(columns=["MSE", "MAE", "R2"]) for name in target_names}

        # 保存新指标
        for col_name in target_names:
            new_row = pd.DataFrame([metrics_dict[col_name]], index=[len(self.history_metrics_multi[col_name])])
            self.history_metrics_multi[col_name] = pd.concat(
                [self.history_metrics_multi[col_name], new_row]
            )

        # 绘制历史图 + 阈值线
        for col_name in target_names:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            fig, ax = plt.subplots(figsize=(9, 3))
            canvas = FigureCanvas(fig)

            hist_df = self.history_metrics_multi[col_name]
            hist_df.plot(ax=ax, marker="o")

            # 阈值线
            ax.axhline(y=mse_limit, color='orange', linestyle='--', label=f"MSE 阈值={mse_limit}")
            ax.axhline(y=r2_limit, color='green', linestyle='--', label=f"R² 阈值={r2_limit}")

            ax.set_title(f"{col_name} 历史指标变化")
            ax.set_xlabel("评估次数")
            ax.set_ylabel("数值")
            ax.legend(fontsize=8)
            layout.addWidget(canvas)
            history_tabs.addTab(tab, col_name)

        QMessageBox.information(self, "评估完成", "模型评估和图表生成完成！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RegressionMonitorGUI()
    gui.show()
    sys.exit(app.exec())
