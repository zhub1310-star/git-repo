# data_processor.py
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def main(config, log_callback=None, progress_callback=None, stop_event=None):
    if log_callback:
        log_callback(">>> data_processor.main() 被调用 ✅", "INFO")

    """
    config keys:
        base_path, excel_path, start_date (YYYY-MM-DD), channel_count,
        pn500 (list), target_items (list), max_sn_count (int), output_dir
    """
    def log(msg, level="INFO"):
        if callable(log_callback):
            try:
                log_callback(msg, level)
            except Exception:
                # fallback to print if callback fails
                print(f"[{level}] {msg}")
        else:
            print(f"[{level}] {msg}")

    # read config with defaults
    BASE_PATH = config.get("base_path")
    EXCEL_PATH = config.get("excel_path")
    START_DATE_STR = config.get("start_date")
    CHANNEL_COUNT = int(config.get("channel_count", 8))
    TARGET_PN500 = set([s.strip() for s in config.get("pn500", []) if s and s.strip()])
    TARGET_ITEM_NAMES = set([s.strip() for s in config.get("target_items", []) if s and s.strip()])
    MAX_SN_COUNT = int(config.get("max_sn_count", 15000))
    OUTPUT_DIR = config.get("output_dir", "dataoutput")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # validate required
    if not BASE_PATH or not EXCEL_PATH or not START_DATE_STR:
        log("配置错误：base_path/excel_path/start_date 必须提供", "ERROR")
        return

    try:
        START_DATE = datetime.strptime(START_DATE_STR, "%Y,%m,%d")
    except Exception as e:
        log(f"开始日期解析失败: {e}", "ERROR")
        return

    # caches
    SN_CACHE = []
    SN_DATA_CACHE = []

    # helpers
    def get_date_path(base_path: str, target_date: datetime) -> str:
        return os.path.join(base_path, target_date.strftime('%Y'), target_date.strftime('%m'), target_date.strftime('%d'))

    def process_condition_data(root: ET.Element):
        temp_data = defaultdict(lambda: {"temps": defaultdict(lambda: {"dut_channels": defaultdict(lambda: {"items": defaultdict(list)})})})
        for condition in root.findall(".//Condition"):
            cond_name = condition.attrib.get('CondName', '')
            if 'FMT' in cond_name:
                Bitrate = condition.attrib.get('Bitrate', 'N/A')
                temp = condition.attrib.get('Temp', 'N/A')
                dut_channel = condition.attrib.get('DutChannel', 'N/A')
                if Bitrate != 'N/A' and temp != 'N/A':
                    dut_channel_data = temp_data[Bitrate]["temps"][temp]["dut_channels"][dut_channel]
                    for fmt in condition.findall('FMT'):
                        for data in fmt.findall('Data'):
                            item_name = data.attrib.get('ItemName', 'N/A')
                            item_value = data.attrib.get('ItemValue', 'N/A')
                            if item_name in TARGET_ITEM_NAMES and item_value != 'N/A':
                                dut_channel_data["items"][item_name].append(item_value)
        return temp_data

    def is_sn_valid(temp_data) -> bool:
        # Ensure BiasDAC exists and in [0,5000]
        for bitrate_data in temp_data.values():
            for temp_info in bitrate_data["temps"].values():
                all_items = defaultdict(list)
                for ch_data in temp_info["dut_channels"].values():
                    for item_name, values in ch_data["items"].items():
                        all_items[item_name].extend(values)
                bias_values = all_items.get("BiasDAC", [])
                if not bias_values:
                    return False
                try:
                    invalid = [v for v in bias_values if not (0 <= float(v) <= 5000)]
                    if invalid:
                        return False
                except Exception:
                    return False
        return True

    def flush_sn_cache():
        """将 SN_DATA_CACHE 写出成 TXT（按原逻辑），然后清空缓存"""
        if not SN_DATA_CACHE:
            return
        # try to get time suffix
        last_time = SN_CACHE[-1][1] if SN_CACHE else None
        try:
            time_part = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S") if last_time else datetime.now().strftime("%Y%m%d_%H%M%S")
        except Exception:
            time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_file = os.path.join(OUTPUT_DIR, f"SN_PN_DATA_{time_part}.txt")

        # collect all temperature numeric values
        all_temps = set()
        for entry in SN_DATA_CACHE:
            for bitrate_data in entry["Data"].values():
                for t in bitrate_data["temps"].keys():
                    try:
                        all_temps.add(float(t))
                    except Exception:
                        continue
        sorted_temps = sorted(all_temps)
        temp_label_map = {}
        if len(sorted_temps) == 1:
            temp_label_map[sorted_temps[0]] = "RT"
        elif len(sorted_temps) == 2:
            temp_label_map[sorted_temps[0]] = "RT"
            temp_label_map[sorted_temps[1]] = "HT"
        elif len(sorted_temps) == 3:
            temp_label_map[sorted_temps[0]] = "LT"
            temp_label_map[sorted_temps[1]] = "RT"
            temp_label_map[sorted_temps[2]] = "HT"

        header = ["SN", "EndTime", "PN", "Bitrate", "Channel"]
        for label in ["HT", "RT", "LT"]:
            header.extend([f"{label}_Temp", f"{label}_BiasDAC", f"{label}_MPDADC", f"{label}_PDADC", f"{label}_TempADC"])

        try:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\t".join(header) + "\n")
                for entry in SN_DATA_CACHE:
                    sn = entry.get("CustomerSN", "")
                    end_time = entry.get("EndTime", "")
                    pn = entry.get("PN500", "")
                    for bitrate, bitrate_data in entry["Data"].items():
                        all_channels = set()
                        for temp_str, temp_info in bitrate_data["temps"].items():
                            all_channels.update(temp_info["dut_channels"].keys())
                        for ch in sorted(all_channels):
                            row = [sn, end_time, pn, bitrate, ch]
                            for label in ["HT", "RT", "LT"]:
                                found = False
                                for temp_str, temp_info in bitrate_data["temps"].items():
                                    try:
                                        temp_float = float(temp_str)
                                    except Exception:
                                        continue
                                    if temp_label_map.get(temp_float) == label:
                                        ch_data = temp_info["dut_channels"].get(ch)
                                        if ch_data:
                                            row.append(temp_str)
                                            for item in TARGET_ITEM_NAMES:
                                                row.append(ch_data["items"].get(item, [""])[0])
                                            found = True
                                            break
                                if not found:
                                    row.extend(["" for _ in range(1 + len(TARGET_ITEM_NAMES))])
                            f.write("\t".join(row) + "\n")
            log(f"完成 TXT 输出：{txt_file}")
        except Exception as e:
            log(f"写 TXT 失败: {e}", "ERROR")
        finally:
            SN_CACHE.clear()
            SN_DATA_CACHE.clear()

    def append_xml_summary_csv(customer_sn, end_time, pn500, temp_data):
        output_path = os.path.join(OUTPUT_DIR, "XML_All_SN_Data.csv")
        write_header = not os.path.exists(output_path)
        rows = []
        for bitrate, bitrate_data in temp_data.items():
            for temp, temp_info in bitrate_data["temps"].items():
                for ch, ch_data in temp_info["dut_channels"].items():
                    row_dict = {
                        "SN": customer_sn,
                        "EndTime": end_time,
                        "PN500": pn500,
                        "Bitrate": bitrate,
                        "Channel": ch,
                        "Temp": temp
                    }

                    for item in TARGET_ITEM_NAMES:
                        row_dict[item] = ch_data["items"].get(item, [""])[0]   
                    rows.append(row_dict)

        try:
            import csv
            target_item_list = list(TARGET_ITEM_NAMES)
            fieldnames = ["SN", "EndTime", "PN500", "Bitrate", "Channel", "Temp"] + target_item_list
            with open(output_path, mode="a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
            log(f"📝 SN 数据追加到 XML 汇总表：{output_path}")
        except Exception as e:
            log(f"写 XML 汇总 CSV 失败: {e}", "ERROR")

    def process_ttr_and_interp(ttr_path: str, heater_csv_path: str) -> bool:
        sn_local = "UnknownSN"
        try:
            with open(ttr_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            try:
                interp_table = pd.read_csv(heater_csv_path)
            except Exception as e:
                log(f"HeaterCurve CSV 读取失败: {heater_csv_path}，原因: {e}", "ERROR")
                return False

            for line in lines:
                if "CustomerSN:" in line:
                    sn_local = line.strip().split("CustomerSN:")[1].strip()

            start_index = end_index = None
            for i, line in enumerate(lines):
                if "======X_HeaterTuneByMpd======" in line:
                    start_index = i
                elif "======TxPwrCheck======" in line and start_index is not None:
                    end_index = i
                    break
            if start_index is None or end_index is None:
                log(f"未找到 HeaterTune 段: {ttr_path}", "WARN")
                return False

            section = lines[start_index:end_index]
            bias_values = []
            heater_dacs = []
            power_values = []
            max_ratios = []
            min_ratios = []
            real_ratios = []
            target_ratios = []
            target_heater_dacs = []

            for line in section:
                if "Set BiasDAC:" in line:
                    bias_values = [int(x) for x in line.split("Set BiasDAC:")[1].strip().split(";") if x != ""]
                elif "HeaterDACs:" in line:
                    heater_dacs = [float(x) for x in line.split("HeaterDACs:")[1].strip().split(";") if x != ""]
                elif "Powers(dBm):" in line:
                    power_values = [float(x) for x in line.split("Powers(dBm):")[1].strip().split(";") if x != ""]
                elif "MaxRatio:" in line:
                    try:
                        max_ratios.append(float(line.split("MaxRatio:")[1].strip()))
                    except:
                        max_ratios.append(np.nan)
                elif "MinRatio:" in line:
                    try:
                        min_ratios.append(float(line.split("MinRatio:")[1].strip()))
                    except:
                        min_ratios.append(np.nan)
                elif "RealRatio:" in line:
                    try:
                        real_ratios.append(float(line.split("RealRatio:")[1].strip()))
                    except:
                        real_ratios.append(np.nan)
                elif "TargetRatio:" in line:
                    try:
                        target_ratios.append(float(line.split("TargetRatio:")[1].strip()))
                    except:
                        target_ratios.append(np.nan)
                elif "TargetHeaterDac: " in line:
                    try:
                        target_heater_dacs.append(float(line.split("TargetHeaterDac:")[1].strip()))
                    except:
                        target_heater_dacs.append(np.nan)

            # basic validation
            if not (len(bias_values) == len(heater_dacs) == len(power_values) == len(max_ratios) == len(min_ratios) == len(real_ratios) == len(target_ratios) == len(target_heater_dacs) == CHANNEL_COUNT):
                log(f"TTR 段数据不完整或通道数不匹配 ({ttr_path})", "WARN")
                return False

            df_ttr = pd.DataFrame({
                "Channel": [f"CH{i+1}" for i in range(CHANNEL_COUNT)],
                "BiasDAC": bias_values,
                "HeaterDAC": heater_dacs,
                "MaxPower": power_values,
                "MaxRatio": max_ratios,
                "MinRatio": min_ratios,
                "RealRatio": real_ratios,
                "TargetRatio": target_ratios,
                "TargetHeaterDac": target_heater_dacs
            })
            df_ttr.insert(0, "SN", sn_local)

            # prepare lists
            mpd_list = []
            pd_list = []
            mpdmax_list = []
            pdmax_list = []
            targetmpd_list = []
            targetpd_list = []

            for idx, row in df_ttr.iterrows():
                ch_label = row['Channel'][-1]
                real_ratio = row['RealRatio']
                target_ratio = row['TargetRatio']

                ratio_col = f"Ratio_{ch_label}"
                mpd_col = f"MpdADC_{ch_label}"
                pd_col = f"PdADC_{ch_label}"

                if ratio_col not in interp_table.columns or mpd_col not in interp_table.columns or pd_col not in interp_table.columns:
                    log(f"插值表缺少列: {ratio_col} / {mpd_col} / {pd_col}，跳过 CH {ch_label}", "WARN")
                    mpd_list.append(np.nan)
                    pd_list.append(np.nan)
                    mpdmax_list.append(np.nan)
                    pdmax_list.append(np.nan)
                    targetmpd_list.append(np.nan)
                    targetpd_list.append(np.nan)
                    continue

                ratio_vals = interp_table[ratio_col].values
                mpd_vals = interp_table[mpd_col].values
                pd_vals = interp_table[pd_col].values

                if len(ratio_vals) < 2 or len(mpd_vals) < 2 or len(pd_vals) < 2:
                    log(f"插值点不足: {heater_csv_path} CH {ch_label}", "WARN")
                    return False

                try:
                    f_mpd = interp1d(ratio_vals, mpd_vals, bounds_error=False, fill_value="extrapolate")
                    f_pd = interp1d(ratio_vals, pd_vals, bounds_error=False, fill_value="extrapolate")
                    mpd_val = float(f_mpd(real_ratio))
                    pd_val = float(f_pd(real_ratio))
                    target_mpd_val = float(f_mpd(target_ratio))
                    target_pd_val = float(f_pd(target_ratio))
                    max_idx = int(np.argmax(ratio_vals))
                    mpd_max = float(mpd_vals[max_idx])
                    pd_max = float(pd_vals[max_idx])
                except Exception as e:
                    log(f"TTR 插值失败 CH {ch_label}: {e}", "WARN")
                    mpd_val = pd_val = target_mpd_val = target_pd_val = mpd_max = pd_max = np.nan

                mpd_list.append(mpd_val)
                pd_list.append(pd_val)
                targetmpd_list.append(target_mpd_val)
                targetpd_list.append(target_pd_val)
                mpdmax_list.append(mpd_max)
                pdmax_list.append(pd_max)

            # append results to CSV
            out_df = df_ttr.copy()
            out_df["RealMPDADC"] = mpd_list
            out_df["RealPDADC"] = pd_list
            out_df["TargetMPDADC"] = targetmpd_list
            out_df["TargetPDADC"] = targetpd_list
            out_df["MAXMPDADC"] = mpdmax_list
            out_df["MAXPDADC"] = pdmax_list

            ttr_output_path = os.path.join(OUTPUT_DIR, "TTR_All_Interp.csv")
            try:
                write_header = not os.path.exists(ttr_output_path)
                out_df.to_csv(ttr_output_path, mode='a', index=False, header=write_header, encoding="utf-8-sig")
                log(f"✅ 插值汇总已追加保存至：{ttr_output_path}")
            except Exception as e:
                log(f"TTR CSV 写入失败: {e}", "ERROR")
                return False

            return True

        except Exception as e:
            log(f"TTR 插值处理异常 ({sn_local}): {e}", "WARN")
            return False

    # --- load SN list from Excel ---
    try:
        df_sn = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        if "SN" not in df_sn.columns:
            log("Excel 必须包含 'SN' 列", "ERROR")
            return
        sn_set = set(df_sn["SN"].astype(str).str.strip())
    except Exception as e:
        log(f"读取 Excel 失败: {e}", "ERROR")
        return

    total_sn = len(sn_set) or 1
    processed = 0
    log(f"📋 目标 SN 数量：{total_sn}", "INFO")
    # --- iterate dates and SNs ---
    current_date = START_DATE
    today = datetime.now().date()
    while current_date.date() <= today:
        if stop_event and stop_event.is_set():
            log("用户请求停止，终止日期循环", "WARN")
            break

        date_path = get_date_path(BASE_PATH, current_date)
        if os.path.exists(date_path):
            # 遍历日期下的所有 SN 文件夹
            for sn in list(sn_set):
                if stop_event and stop_event.is_set():
                    log("用户请求停止，退出 SN 循环", "WARN")
                    break

                sn_folder = os.path.join(date_path, sn)
                if not os.path.exists(sn_folder):
                    continue

                # 遍历 SN 文件夹里的子文件夹（保证可以处理多层 SN 文件夹结构）
                for root, dirs, files in os.walk(sn_folder):
                    if stop_event and stop_event.is_set():
                        log("用户请求停止，退出 os.walk 循环", "WARN")
                        break

                    xml_file = os.path.join(root, "TestData.xml")
                    ttr_file = os.path.join(root, "Ttr.txt")
                    heater_file = os.path.join(root, "HeaterCurve.csv")
                    #print(f"  xml_file: {os.path.exists(xml_file)}, ttr_file: {os.path.exists(ttr_file)}, heater_file: {os.path.exists(heater_file)}")
                    if stop_event and stop_event.is_set():
                        log("用户请求停止，在文件检查前退出", "WARN")
                        break

                    if os.path.exists(xml_file) and os.path.exists(ttr_file) and os.path.exists(heater_file):
                        
                        ok_ttr = process_ttr_and_interp(ttr_file, heater_file)
                        if stop_event and stop_event.is_set():
                            log("用户请求停止，在 TTR 插值后退出", "WARN")
                            break
                        if not ok_ttr:
                            continue

                        # 解析 XML
                        try:
                            tree = ET.parse(xml_file)
                            root_xml = tree.getroot()
                            pn500 = root_xml.attrib.get("PN500", "N/A")
                            category = root_xml.attrib.get("Category", "N/A")
                            if category != "PROD" or (TARGET_PN500 and pn500 not in TARGET_PN500):
                                continue

                            if stop_event and stop_event.is_set():
                                log("用户请求停止，在 XML 解析前退出", "WARN")
                                break

                            for record in root_xml.findall(".//Record"):
                                if stop_event and stop_event.is_set():
                                    log("用户请求停止，在 Record 循环中退出", "WARN")
                                    break

                                if record.attrib.get("MesFailCode", "N/A") == "良品":
                                    customer_sn = record.attrib.get("CustomerSN", "N/A")
                                    end_time = record.attrib.get("EndTime", "N/A")
                                    temp_data = process_condition_data(root_xml)
                                    if not is_sn_valid(temp_data):
                                        log(f"SN {customer_sn} 不满足校验条件，跳过", "WARN")
                                        continue

                                    SN_CACHE.append((customer_sn, end_time))
                                    SN_DATA_CACHE.append({
                                        "CustomerSN": customer_sn,
                                        "EndTime": end_time,
                                        "PN500": pn500,
                                        "Data": temp_data
                                    })

                                    append_xml_summary_csv(customer_sn, end_time, pn500, temp_data)
                                    log(f"✅ 已收集 SN: {customer_sn}")

                                    processed += 1
                                    if callable(progress_callback):
                                        try:
                                            progress_callback(min(1.0, processed / total_sn))
                                        except Exception:
                                            pass

                                    # flush if full
                                    if len(SN_CACHE) >= MAX_SN_COUNT:
                                        flush_sn_cache()

                                    break  # 每个 SN 只处理一次
                        except Exception as e:
                            log(f"解析 XML 失败: {xml_file}，错误: {e}", "ERROR")
        # 下一天
        current_date += timedelta(days=1)

    # flush 剩余缓存
    if SN_DATA_CACHE:
        flush_sn_cache()
    log("处理完成。", "INFO")

