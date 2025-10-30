# main_gui.py
import threading
import os
import time
from datetime import datetime
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import data_processor

# appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class MonitorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("工业监控 — SN 数据处理")
        self.geometry("1350x800")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._stop_event = threading.Event()
        self._worker_thread = None

        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        left = ctk.CTkFrame(container, corner_radius=6, fg_color="#232323")
        left.place(relx=0, rely=0, relwidth=0.36, relheight=1)

        right = ctk.CTkFrame(container, corner_radius=6, fg_color="#1a1a1a")
        right.place(relx=0.37, rely=0, relwidth=0.63, relheight=1)

        # LEFT controls
        header = ctk.CTkLabel(left, text="参数配置", font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(anchor="nw", padx=12, pady=(12,6))

        padx=12; pady=6

        self.base_path_var = ctk.StringVar(value=r"\\thnas01\THNAS01\MFG\Module Data\ATSData")
        ctk.CTkLabel(left, text="ATS 数据文件夹：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.base_path_var, width=360).pack(anchor="nw", padx=padx)
        ctk.CTkButton(left, text="选择文件夹", width=140, command=self._choose_base).pack(anchor="nw", padx=padx, pady=(6,6))

        self.excel_path_var = ctk.StringVar(value=r"C:\Users\hongbin.zhu\Desktop\TTRSN.xlsx")
        ctk.CTkLabel(left, text="目标 SN Excel（含 SN 列）：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.excel_path_var, width=360).pack(anchor="nw", padx=padx)
        ctk.CTkButton(left, text="选择文件", width=140, command=self._choose_excel).pack(anchor="nw", padx=padx, pady=(6,6))

        self.start_date_var = ctk.StringVar(value=datetime.now().strftime("%Y,%m,%d"))
        ctk.CTkLabel(left, text="开始日期 (YYYY,MM,DD)：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.start_date_var, width=200).pack(anchor="nw", padx=padx, pady=(0,6))

        self.channel_count_var = ctk.StringVar(value="8")
        ctk.CTkLabel(left, text="通道数量 (channel count)：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.channel_count_var, width=120).pack(anchor="nw", padx=padx, pady=(0,6))

        self.pn_var = ctk.StringVar(value="562-0431")
        ctk.CTkLabel(left, text="PN500（逗号分隔）：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.pn_var, width=360).pack(anchor="nw", padx=padx, pady=(0,6))

        self.items_var = ctk.StringVar(value="BiasDAC,MPDADC,PDADC,TempADC")
        ctk.CTkLabel(left, text="目标 Item 名称（逗号分隔）：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.items_var, width=360).pack(anchor="nw", padx=padx, pady=(0,6))

        self.max_sn_var = ctk.StringVar(value="15000")
        ctk.CTkLabel(left, text="最大 SN 数量：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.max_sn_var, width=120).pack(anchor="nw", padx=padx, pady=(0,6))

        self.output_dir_var = ctk.StringVar(value="dataoutput")
        ctk.CTkLabel(left, text="输出目录：").pack(anchor="nw", padx=padx, pady=(pady,0))
        ctk.CTkEntry(left, textvariable=self.output_dir_var, width=360).pack(anchor="nw", padx=padx)
        ctk.CTkButton(left, text="选择输出目录", width=140, command=self._choose_output).pack(anchor="nw", padx=padx, pady=(6,12))

        # bottom buttons
        btn_frame = ctk.CTkFrame(left, fg_color="#101010")
        btn_frame.pack(anchor="s", side="bottom", fill="x", padx=padx, pady=12)

        self.start_btn = ctk.CTkButton(btn_frame, text="▶ 开始", fg_color="#0a84ff", hover_color="#0063c6", command=self._on_start)
        self.start_btn.pack(side="left", padx=10, pady=8)
        self.stop_btn = ctk.CTkButton(btn_frame, text="■ 停止", fg_color="#ff3b30", hover_color="#cc2b24", command=self._on_stop)
        self.stop_btn.pack(side="left", padx=10, pady=8)
        self.open_out_btn = ctk.CTkButton(btn_frame, text="📂 打开输出目录", command=self._open_output)
        self.open_out_btn.pack(side="right", padx=10, pady=8)

        # RIGHT: log & status
        top_status = ctk.CTkFrame(right, fg_color="#0f0f0f")
        top_status.pack(fill="x", padx=12, pady=12)

        self.status_label = ctk.CTkLabel(top_status, text="状态：空闲", anchor="w", font=ctk.CTkFont(size=13, weight="bold"))
        self.status_label.pack(side="left", padx=6)

        self.progress_bar = ctk.CTkProgressBar(top_status, width=360)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(side="right", padx=6)

        log_label = ctk.CTkLabel(right, text="运行日志", anchor="w", font=ctk.CTkFont(size=13, weight="bold"))
        log_label.pack(anchor="nw", padx=12)
        self.log_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, bg="#0a0a0a", fg="#e6e6e6", font=("Consolas", 11))
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(6,12))
        self.log_text.tag_configure("INFO", foreground="#9be7a3")
        self.log_text.tag_configure("WARN", foreground="#ffc66b")
        self.log_text.tag_configure("ERROR", foreground="#ff6b6b")
        self.log_text.tag_configure("DEBUG", foreground="#9fb6ff")

        self._total_sn = 0
        self._processed_sn = 0

    # helpers
    def _choose_base(self):
        p = filedialog.askdirectory()
        if p:
            self.base_path_var.set(p)

    def _choose_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir_var.set(p)

    def _choose_excel(self):
        p = filedialog.askopenfilename(filetypes=[("Excel 文件", "*.xlsx;*.xls")])
        if p:
            self.excel_path_var.set(p)

    def _open_output(self):
        out = self.output_dir_var.get()
        if not out:
            messagebox.showinfo("提示", "请先指定输出目录")
            return
        if os.path.exists(out):
            try:
                os.startfile(os.path.abspath(out))
            except Exception as e:
                messagebox.showerror("错误", f"无法打开目录: {e}")
        else:
            messagebox.showinfo("提示", "输出目录不存在")

    def _append_log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%Y,%m,%d %H:%M:%S")
        tag = level if level in ("INFO","WARN","ERROR","DEBUG") else "INFO"
        try:
            self.log_text.insert("end", f"[{ts}] {msg}\n", tag)
        except Exception:
            self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    # callbacks passed to data_processor
    def _log_cb(self, msg, level="INFO"):
        self.after(0, self._append_log, msg, level)

    def _progress_cb(self, fraction):
        self.after(0, self._set_progress, fraction)

    def _set_progress(self, fraction):
        frac = max(0.0, min(1.0, float(fraction)))
        self.progress_bar.set(frac)
        self._processed_sn = int(round(frac * (self._total_sn or 1)))
        self.status_label.configure(text=f"状态：运行中（{self._processed_sn}/{self._total_sn}）")

    def _on_start(self):
        base = self.base_path_var.get().strip()
        excel = self.excel_path_var.get().strip()
        start_date = self.start_date_var.get().strip()
        try:
            datetime.strptime(start_date, "%Y,%m,%d")
        except Exception:
            messagebox.showerror("错误", "开始日期格式应为 YYYY,MM,DD")
            return
        try:
            channel_count = int(self.channel_count_var.get().strip() or "8")
            if channel_count <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("错误", "通道数量请输入正整数")
            return

        try:
            max_sn = int(self.max_sn_var.get().strip() or "15000")
        except Exception:
            max_sn = 15000


        items = [i.strip() for i in self.items_var.get().split(",") if i.strip()]
        pns = [i.strip() for i in self.pn_var.get().split(",") if i.strip()]
        try:
            max_sn = int(self.max_sn_var.get())
        except Exception:
            max_sn = 15000
        outdir = self.output_dir_var.get().strip() or "dataoutput"
        os.makedirs(outdir, exist_ok=True)

        if not os.path.exists(base):
            messagebox.showerror("错误", "ATS 数据文件夹不存在")
            return
        if not os.path.isfile(excel):
            messagebox.showerror("错误", "Excel 文件不存在")
            return

        # compute total SN for progress baseline
        try:
            df = pd.read_excel(excel, engine='openpyxl')
            if "SN" not in df.columns:
                messagebox.showerror("错误", "Excel 必须包含 'SN' 列")
                return
            sn_set = set(df["SN"].astype(str).str.strip())
        except Exception as e:
            messagebox.showerror("错误", f"读取 Excel 失败：{e}")
            return

        self._total_sn = len(sn_set)
        self._processed_sn = 0
        self.progress_bar.set(0.0)
        self.status_label.configure(text="状态：准备运行")
        self.log_text.delete('1.0', 'end')

        config = {
            "base_path": base,
            "excel_path": excel,
            "start_date": start_date,
            "channel_count": channel_count,
            "pn500": pns,
            "target_items": items,
            "max_sn_count": max_sn,
            "output_dir": outdir
        }

        # start worker
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_main, args=(config,), daemon=True)
        self._worker_thread.start()
        self.start_btn.configure(state="disabled")
        self._append_log("▶ 启动后台处理线程...", "INFO")
        self.status_label.configure(text=f"状态：运行中（0/{self._total_sn}）")

    def _on_stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            self._append_log("■ 已请求停止，后台线程将尽快停止...", "WARN")
            self.start_btn.configure(state="normal")
        else:
            self._append_log("■ 没有正在运行的后台任务", "WARN")

    def _worker_main(self, config):
        try:
            def log_cb(msg, level="INFO"):
                self._log_cb(msg, level)
            def progress_cb(frac):
                self._progress_cb(frac)

            data_processor.main(config, log_callback=log_cb, progress_callback=progress_cb, stop_event=self._stop_event)
            if not self._stop_event.is_set():
                self._append_log("✔ 后台任务完成。", "INFO")
                self.after(0, lambda: self.status_label.configure(text="状态：已完成"))
            else:
                self._append_log("✔ 后台任务已停止（用户请求）。", "WARN")
                self.after(0, lambda: self.status_label.configure(text="状态：已停止"))
        except Exception as e:
            self._append_log(f"❌ 后台线程异常: {e}", "ERROR")
            self.after(0, lambda: self.status_label.configure(text="状态：异常"))
        finally:
            self.after(0, lambda: self.start_btn.configure(state="normal"))

    def _on_close(self):
        if self._worker_thread and self._worker_thread.is_alive():
            if messagebox.askyesno("确认", "后台任务正在运行，确定退出并停止任务？"):
                self._stop_event.set()
                # give thread brief time to stop
                time.sleep(0.2)
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    try:
        import pandas as pd  # ensure dependency available at start
    except Exception:
        messagebox.showerror("依赖缺失", "请先安装 pandas: pip install pandas")
        raise

    app = MonitorGUI()
    app.mainloop()
