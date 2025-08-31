#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIGS IV Data Packet Analysis Tool

raw_data.txt から16進数列を読み込み、CIGS IV 測定パケットを解析します。

パケット構造（想定）:
- 1パケット = 64バイト
- 最終バイト = CRC8 (XORベース, 先頭〜63番目までのXOR)
- 先頭バイトが 0xFF の場合のみ、以下のヘッダ＋環境データを含む:
  - START_MARKER: 1B (0xFF)
  - TIME: 4B (BE)
  - 環境データ (6B): PD(12bit), Temp_PY_TOP(12bit), Temp_PY_BOT(12bit), Temp_MIS7(12bit)
- データ部: 3Bで2つの12bit値（上位=電圧, 下位=電流）を格納

出力先:
- 同一フォルダ配下の output/ にCSVやJSONなどを保存
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd


@dataclass
class PacketHeader:
    """パケットヘッダ情報"""

    start_marker: Optional[int]  # 最初のパケットのみ 0xFF
    time: int                    # 4バイトのタイムスタンプ
    pd: int                      # PD値 (12bit)
    temp_py_top: int             # 温度PY_TOP (12bit)
    temp_py_bot: int             # 温度PY_BOT (12bit)
    temp_mis7: int               # 温度MIS7 (12bit)


@dataclass
class IVDataStep:
    """IV測定データステップ（12bit値x2）"""

    data0: int  # 12bit (電圧)
    data1: int  # 12bit (電流)


class PacketAnalyzer:
    def __init__(self, filename: Optional[str] = None, analysis_mode: str = "iv_measurement", *, check_crc: bool = False) -> None:
        """解析クラスの初期化

        Args:
            filename: 入力ファイルパス。未指定(None)ならモードに応じた既定ファイルを使用。
            analysis_mode: "iv_measurement" | "environment" | "piclog"
        """

        # ベースディレクトリ（このスクリプトのある場所）
        self.base_dir = Path(__file__).resolve().parent

        # モード正当性
        self.analysis_mode = analysis_mode
        if analysis_mode not in ["iv_measurement", "environment", "piclog"]:
            raise ValueError("analysis_mode must be 'iv_measurement', 'environment', or 'piclog'")

        # モードに応じた既定ファイル名
        default_map = {
            "iv_measurement": "iv_data.txt",
            "environment": "env_data.txt",
            "piclog": "piclog_data.txt",
        }

        # ファイル名の決定（None または空文字なら既定を採用）
        chosen = (filename or "").strip() if filename is not None else ""
        if not chosen:
            chosen = default_map[self.analysis_mode]
        self.filename = chosen

        # 入力パスを決定（絶対はそのまま、相対はベースディレクトリ基準）
        in_path = Path(self.filename)
        self.input_path = in_path if in_path.is_absolute() else (self.base_dir / in_path)

        # 出力先
        self.output_dir = self.base_dir / "output"

        # 解析用状態
        self.raw_data: List[str] = []
        self.PACKET_SIZE = 64
        self.START_MARKER = 0xFF
        # CRC検証はデフォルト無効（ユーザー要望）
        self.check_crc = check_crc

        # 工学値変換用パラメータ（必要に応じて調整）
        self.VREF = 2.5
        self.ADC_MAX = 4095
        self.CURRENT_SHUNT_R = 0.1
        self.CURRENT_VREF = 1.24
        self.CURRENT_GAIN = 1.0
        self.LMT84_V0 = 1.8639  # 0℃の出力電圧 (V)
        self.LMT84_TC = 0.0055  # 5.5mV/℃

    # --- 工学値変換ヘルパー ---
    def _adc_to_voltage(self, adc_value: int) -> float:
        try:
            return (adc_value / self.ADC_MAX) * self.VREF
        except Exception:
            return float("nan")

    def _adc_to_current(self, adc_value: int) -> float:
        try:
            v = (adc_value / self.ADC_MAX) * self.VREF
            return (v - self.CURRENT_VREF) / self.CURRENT_SHUNT_R / self.CURRENT_GAIN
        except Exception:
            return float("nan")

    def _adc_to_temperature_lmt84(self, adc_value: int) -> float:
        """温度センサ式（ユーザー提供の二次式）で温度[℃]を算出。

        Excel式:
        =(5.506-SQRT(POWER(-5.506,2)+4*0.00176*(870.6-B3)))/(2*(-0.00176))+30
        B3: 電圧[mV]
        """
        try:
            # ADC -> 電圧[V] -> mV
            v_mV = (adc_value / self.ADC_MAX) * self.VREF * 1000.0
            a = -0.00176
            b = -5.506
            c = 870.6 - v_mV
            disc = b * b - 4 * a * c  # = b^2 + 4*0.00176*(870.6 - v_mV)
            if disc < 0:
                disc = 0.0
            import math
            sqrt_disc = math.sqrt(disc)
            temp = (5.506 - sqrt_disc) / (2 * a) + 30.0
            return temp
        except Exception:
            return float("nan")

    def _pd_to_voltage(self, pd_8bit: int) -> float:
        # PDは8bit前提 -> 12bitスケールに拡張してから電圧換算
        try:
            adc12 = int(pd_8bit) * 16
            return self._adc_to_voltage(adc12)
        except Exception:
            return float("nan")

    def read_raw_data(self) -> List[str]:
        """raw_data.txt から64バイト単位のパケット文字列に再構成して返す"""
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                content = f.read()

            hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", content)
            print(f"読み込み完了: {len(hex_pairs)} バイト相当のHEXを検出")

            packets: List[str] = []
            for i in range(0, len(hex_pairs), self.PACKET_SIZE):
                pkt = hex_pairs[i : i + self.PACKET_SIZE]
                if len(pkt) == self.PACKET_SIZE:
                    packets.append(" ".join(pkt))

            self.raw_data = packets
            print(f"64バイト完全パケット数: {len(self.raw_data)}")
            return self.raw_data
        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません -> '{self.input_path}'")
            return []
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return []

    @staticmethod
    def calc_crc8(frame: List[int]) -> int:
        """XORベースのCRC8（先頭から順次XOR）"""
        if not frame:
            return 0
        crc = frame[0]
        for b in frame[1:]:
            crc ^= b
        return crc & 0xFF

    def parse_packet_header(self, packet_bytes: List[int]) -> Tuple[PacketHeader, int]:
        """パケット先頭からヘッダ情報を復元し、ヘッダ長を返す

        戻り値: (header, header_size)
        """
        idx = 0
        start_marker: Optional[int] = None

        if packet_bytes[0] == self.START_MARKER:
            start_marker = packet_bytes[idx]
            idx += 1
            # TIME (4B, BE)
            time = (
                (packet_bytes[idx] << 24)
                | (packet_bytes[idx + 1] << 16)
                | (packet_bytes[idx + 2] << 8)
                | packet_bytes[idx + 3]
            )
            idx += 4

            # 環境データ 6B = 12bit x 4 値（PD, TOP, BOT, MIS7）
            pd = (packet_bytes[idx] << 4) | ((packet_bytes[idx + 1] >> 4) & 0x0F)
            temp_py_top = ((packet_bytes[idx + 1] & 0x0F) << 8) | packet_bytes[idx + 2]
            temp_py_bot = (packet_bytes[idx + 3] << 4) | ((packet_bytes[idx + 4] >> 4) & 0x0F)
            temp_mis7 = ((packet_bytes[idx + 4] & 0x0F) << 8) | packet_bytes[idx + 5]
            idx += 6
        else:
            time = 0
            pd = 0
            temp_py_top = 0
            temp_py_bot = 0
            temp_mis7 = 0

        header = PacketHeader(
            start_marker=start_marker,
            time=time,
            pd=pd,
            temp_py_top=temp_py_top,
            temp_py_bot=temp_py_bot,
            temp_mis7=temp_mis7,
        )
        return header, idx

    def parse_iv_data(self, packet_bytes: List[int], start_idx: int, end_idx: int) -> List[IVDataStep]:
        """IV測定データを復元

        3バイトで2つの12bit値（上位=電圧, 下位=電流）。end_idx は排他的。
        範囲外や端数は自動で無視。
        """
        iv_data: List[IVDataStep] = []
        if start_idx is None or end_idx is None:
            return iv_data
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(packet_bytes):
            end_idx = len(packet_bytes)
        if start_idx >= end_idx:
            return iv_data

        available = end_idx - start_idx
        steps = available // 3
        for s in range(steps):
            i = start_idx + s * 3
            b0 = packet_bytes[i]
            b1 = packet_bytes[i + 1]
            b2 = packet_bytes[i + 2]
            voltage_12 = ((b0 << 4) | ((b1 >> 4) & 0x0F)) & 0x0FFF
            current_12 = (((b1 & 0x0F) << 8) | b2) & 0x0FFF
            iv_data.append(IVDataStep(data0=voltage_12, data1=current_12))
        return iv_data

    def parse_environment_data(self, packet_bytes: List[int], start_idx: int, end_idx: int) -> List[Dict[str, int]]:
        """環境データ（6B単位）を復元（Cの logdata[6] 形式）"""
        env_data: List[Dict[str, int]] = []
        if start_idx is None or end_idx is None:
            return env_data
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(packet_bytes):
            end_idx = len(packet_bytes)
        if start_idx >= end_idx:
            return env_data

        available = end_idx - start_idx
        records = available // 6
        for r in range(records):
            i = start_idx + r * 6
            logdata = packet_bytes[i : i + 6]
            temp_top = (logdata[0] << 4) | ((logdata[1] >> 4) & 0x0F)
            temp_bot = ((logdata[1] & 0x0F) << 8) | logdata[2]
            temp_mis7 = (logdata[3] << 4) | ((logdata[4] >> 4) & 0x0F)
            pd = logdata[5]
            env_data.append(
                {
                    "temp_top": temp_top,
                    "temp_bot": temp_bot,
                    "temp_mis7": temp_mis7,
                    "pd": pd,
                }
            )
        return env_data

    def analyze_packets(self) -> None:
        """全パケットを走査して概要を表示"""
        if not self.raw_data:
            print("データが未読です。先に read_raw_data() を実行してください。")
            return

        print("=" * 80)
        print("CIGS IV データパケット解析レポート")
        print("=" * 80)

        valid_packets = 0
        invalid_packets = 0

        for i, packet_line in enumerate(self.raw_data, start=1):
            print(f"\nパケット {i}:")
            print(f"Raw: {packet_line}")

            hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
            if len(hex_pairs) != self.PACKET_SIZE:
                print("  エラー: パケットサイズ不正")
                invalid_packets += 1
                continue

            packet_bytes = [int(h, 16) for h in hex_pairs]

            # CRC（要求によりデフォルトではスキップ）
            if self.check_crc:
                payload = packet_bytes[:-1]
                received_crc = packet_bytes[-1]
                calculated_crc = self.calc_crc8(payload)
                crc_ok = received_crc == calculated_crc
                print(
                    f"  CRC検証: {'✓ OK' if crc_ok else '✗ NG'} (rx=0x{received_crc:02X}, calc=0x{calculated_crc:02X})"
                )
                if crc_ok:
                    valid_packets += 1
                else:
                    invalid_packets += 1
            else:
                print("  CRC検証: スキップ（未実施）")

            # ヘッダ
            header, header_size = self.parse_packet_header(packet_bytes)
            print(f"  ヘッダ長: {header_size}B, TIME: 0x{header.time:08X}")
            if header.start_marker == self.START_MARKER:
                print(
                    "  環境: "
                    f"PD={header.pd} (0x{header.pd:03X}), "
                    f"TOP={header.temp_py_top} (0x{header.temp_py_top:03X}), "
                    f"BOT={header.temp_py_bot} (0x{header.temp_py_bot:03X}), "
                    f"MIS7={header.temp_mis7} (0x{header.temp_mis7:03X})"
                )
            else:
                print("  継続パケット（環境なし）")

            # IVデータ
            data_start = header_size
            data_end = self.PACKET_SIZE - 1  # CRC直前まで
            iv = self.parse_iv_data(packet_bytes, data_start, data_end)
            print(f"  IVステップ数: {len(iv)}")
            for j, step in enumerate(iv[:3], start=1):
                print(
                    f"    step{j}: V={step.data0} (0x{step.data0:03X}), I={step.data1} (0x{step.data1:03X})"
                )
            if len(iv) > 3:
                print(f"    ... (残り {len(iv)-3} ステップ)")

        print("\n" + "=" * 80)
        print("解析結果サマリ")
        print("=" * 80)
        print(f"総パケット: {len(self.raw_data)}")
        if self.check_crc:
            print(f"CRC一致: {valid_packets}")
            print(f"CRC不一致: {invalid_packets}")
            if self.raw_data:
                print(f"成功率: {valid_packets/len(self.raw_data)*100:.1f}%")
        else:
            print("CRCサマリ: スキップ（未実施）")

    def export_environment_data_csv(self, output_filename: str = "environment_data.csv") -> None:
        """環境データをCSVに保存（警告: CRC不一致でも継続）"""
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_dir / output_filename
            if not Path(output_filename).is_absolute()
            else Path(output_filename)
        )

        record_counter = 0
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # 要望: 工学値4列のみ
            writer.writerow([
                "Temp_PY_TOP_C",
                "Temp_PY_BOT_C",
                "Temp_MIS7_C",
                "PD_Voltage_V",
            ])

            for packet_idx, packet_line in enumerate(self.raw_data, start=1):
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                packet_bytes = [int(h, 16) for h in hex_pairs]

                if self.check_crc:
                    payload = packet_bytes[:-1]
                    received_crc = packet_bytes[-1]
                    calculated_crc = self.calc_crc8(payload)
                    if received_crc != calculated_crc:
                        print(f"  警告: パケット{packet_idx} CRC不一致 - 出力継続")

                env_records = self.parse_environment_data(packet_bytes, 0, self.PACKET_SIZE - 1)
                for env in env_records:
                    record_counter += 1
                    # 工学値に変換
                    t_top_c = self._adc_to_temperature_lmt84(env["temp_top"])
                    t_bot_c = self._adc_to_temperature_lmt84(env["temp_bot"])
                    t_mis7_c = self._adc_to_temperature_lmt84(env["temp_mis7"])
                    pd_v = self._pd_to_voltage(env["pd"])
                    # 要望: 4列のみ出力
                    writer.writerow([t_top_c, t_bot_c, t_mis7_c, pd_v])

        print(f"環境データを '{output_path}' に保存しました（{record_counter}件）")

    def export_parsed_data(self, output_filename: str = "parsed_iv_data.csv") -> None:
        """IVデータ＋環境ヘッダ情報をCSVに保存（測定セッション継続対応）"""
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_dir / output_filename
            if not Path(output_filename).is_absolute()
            else Path(output_filename)
        )

        valid_count = 0
        sample_number = 0
        current_timestamp = 0
        current_pd = 0
        current_temp_py_top = 0
        current_temp_py_bot = 0
        current_temp_mis7 = 0
        global_step_counter = 0

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Packet",
                "PacketType",
                "SampleNumber",
                "Timestamp",
                "PD",
                "Temp_PY_TOP",
                "Temp_PY_BOT",
                "Temp_MIS7",
                "Step",
                "Data0",
                "Data1",
                # 工学値列
                "Voltage_V",
                "Current_A",
                "Power_W",
                "Temp_PY_TOP_C",
                "Temp_PY_BOT_C",
                "Temp_MIS7_C",
                "PD_Voltage_V",
            ])

            for packet_line in self.raw_data:
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                packet_bytes = [int(h, 16) for h in hex_pairs]

                if self.check_crc:
                    payload = packet_bytes[:-1]
                    received_crc = packet_bytes[-1]
                    calculated_crc = self.calc_crc8(payload)
                    if received_crc != calculated_crc:
                        print(f"  警告: CSV出力パケット{valid_count + 1} CRC不一致 - 処理継続")

                valid_count += 1

                is_ff = packet_bytes[0] == self.START_MARKER
                pkt_type = "FF_Header" if is_ff else "IV_Continuation"

                header, header_size = self.parse_packet_header(packet_bytes)
                if is_ff:
                    sample_number += 1
                    current_timestamp = header.time
                    current_pd = header.pd
                    current_temp_py_top = header.temp_py_top
                    current_temp_py_bot = header.temp_py_bot
                    current_temp_mis7 = header.temp_mis7

                iv = self.parse_iv_data(packet_bytes, header_size, self.PACKET_SIZE - 1)
                for step in iv:
                    global_step_counter += 1
                    # 工学値
                    v = self._adc_to_voltage(step.data0)
                    i = self._adc_to_current(step.data1)
                    p = v * i
                    t_top_c = self._adc_to_temperature_lmt84(current_temp_py_top)
                    t_bot_c = self._adc_to_temperature_lmt84(current_temp_py_bot)
                    t_mis7_c = self._adc_to_temperature_lmt84(current_temp_mis7)
                    pd_v = self._pd_to_voltage(current_pd)
                    writer.writerow(
                        [
                            valid_count,
                            pkt_type,
                            sample_number,
                            current_timestamp,
                            current_pd,
                            current_temp_py_top,
                            current_temp_py_bot,
                            current_temp_mis7,
                            global_step_counter,
                            step.data0,
                            step.data1,
                            v,
                            i,
                            p,
                            t_top_c,
                            t_bot_c,
                            t_mis7_c,
                            pd_v,
                        ]
                    )

        print(
            f"解析結果を '{output_path}' に保存しました（packets={valid_count}, steps={global_step_counter}）"
        )

    def export_packet_log_csv(self, output_filename: str = "piclog_packets.csv") -> None:
        """PICログ用途の簡易CSV（HEXとCRC一致フラグ）"""
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_dir / output_filename
            if not Path(output_filename).is_absolute()
            else Path(output_filename)
        )

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "CRC_Valid", "Hex"])

            for idx, packet_line in enumerate(self.raw_data, start=1):
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                if self.check_crc:
                    bytes_int = [int(h, 16) for h in hex_pairs]
                    received = bytes_int[-1]
                    calculated = self.calc_crc8(bytes_int[:-1])
                    crc_valid = received == calculated
                else:
                    crc_valid = "skipped"
                writer.writerow([idx, crc_valid, " ".join(hex_pairs)])

        print(f"PICログを '{output_path}' に保存しました")

    def build_parsed_dataframe(self) -> pd.DataFrame:
        """解析済みデータをDataFrameで返す（CSV出力はしない）"""
        if not self.raw_data:
            self.read_raw_data()
        if self.analysis_mode == "environment":
            return self._build_environment_dataframe()
        else:
            return self._build_iv_measurement_dataframe()

    def _build_environment_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, int]] = []
        try:
            record_counter = 0
            for packet_idx, packet_line in enumerate(self.raw_data, start=1):
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                packet_bytes = [int(h, 16) for h in hex_pairs]

                if self.check_crc:
                    payload = packet_bytes[:-1]
                    received_crc = packet_bytes[-1]
                    calculated_crc = self.calc_crc8(payload)
                    if received_crc != calculated_crc:
                        print(
                            f"  警告: パケット{packet_idx} CRC不一致 (rx=0x{received_crc:02X}, calc=0x{calculated_crc:02X})"
                        )

                env_records = self.parse_environment_data(packet_bytes, 0, self.PACKET_SIZE - 1)
                for env in env_records:
                    record_counter += 1
                    rows.append(
                        {
                            "PacketIndex": packet_idx,
                            "RecordNumber": record_counter,
                            "Temp_PY_TOP": env["temp_top"],
                            "Temp_PY_BOT": env["temp_bot"],
                            "Temp_MIS7": env["temp_mis7"],
                            "PD": env["pd"],
                            # 工学値
                            "Temp_PY_TOP_C": self._adc_to_temperature_lmt84(env["temp_top"]),
                            "Temp_PY_BOT_C": self._adc_to_temperature_lmt84(env["temp_bot"]),
                            "Temp_MIS7_C": self._adc_to_temperature_lmt84(env["temp_mis7"]),
                            "PD_Voltage_V": self._pd_to_voltage(env["pd"]),
                        }
                    )
            columns = [
                "PacketIndex",
                "RecordNumber",
                "Temp_PY_TOP",
                "Temp_PY_BOT",
                "Temp_MIS7",
                "PD",
                # 工学値
                "Temp_PY_TOP_C",
                "Temp_PY_BOT_C",
                "Temp_MIS7_C",
                "PD_Voltage_V",
            ]
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            print(f"環境DataFrame構築エラー: {e}")
            return pd.DataFrame()

    def _build_iv_measurement_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, int]] = []
        try:
            valid_count = 0
            sample_number = 0
            current_timestamp = 0
            current_pd = 0
            current_temp_py_top = 0
            current_temp_py_bot = 0
            current_temp_mis7 = 0
            global_step_counter = 0

            for packet_line in self.raw_data:
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                packet_bytes = [int(h, 16) for h in hex_pairs]

                if self.check_crc:
                    payload = packet_bytes[:-1]
                    received_crc = packet_bytes[-1]
                    calculated_crc = self.calc_crc8(payload)
                    if received_crc != calculated_crc:
                        print(
                            f"  警告: CRC不一致 (rx=0x{received_crc:02X}, calc=0x{calculated_crc:02X}) - 継続"
                        )

                valid_count += 1
                is_ff = packet_bytes[0] == self.START_MARKER
                pkt_type = "FF_Header" if is_ff else "IV_Continuation"

                header, header_size = self.parse_packet_header(packet_bytes)
                if is_ff:
                    sample_number += 1
                    current_timestamp = header.time
                    current_pd = header.pd
                    current_temp_py_top = header.temp_py_top
                    current_temp_py_bot = header.temp_py_bot
                    current_temp_mis7 = header.temp_mis7

                iv = self.parse_iv_data(packet_bytes, header_size, self.PACKET_SIZE - 1)
                for step in iv:
                    global_step_counter += 1
                    v = self._adc_to_voltage(step.data0)
                    i = self._adc_to_current(step.data1)
                    p = v * i
                    rows.append(
                        {
                            "Packet": valid_count,
                            "PacketType": pkt_type,
                            "SampleNumber": sample_number,
                            "Timestamp": current_timestamp,
                            "PD": current_pd,
                            "Temp_PY_TOP": current_temp_py_top,
                            "Temp_PY_BOT": current_temp_py_bot,
                            "Temp_MIS7": current_temp_mis7,
                            "Step": global_step_counter,
                            "Data0": step.data0,
                            "Data1": step.data1,
                            # 工学値
                            "Voltage_V": v,
                            "Current_A": i,
                            "Power_W": p,
                            "Temp_PY_TOP_C": self._adc_to_temperature_lmt84(current_temp_py_top),
                            "Temp_PY_BOT_C": self._adc_to_temperature_lmt84(current_temp_py_bot),
                            "Temp_MIS7_C": self._adc_to_temperature_lmt84(current_temp_mis7),
                            "PD_Voltage_V": self._pd_to_voltage(current_pd),
                        }
                    )

            columns = [
                "Packet",
                "PacketType",
                "SampleNumber",
                "Timestamp",
                "PD",
                "Temp_PY_TOP",
                "Temp_PY_BOT",
                "Temp_MIS7",
                "Step",
                "Data0",
                "Data1",
                # 工学値
                "Voltage_V",
                "Current_A",
                "Power_W",
                "Temp_PY_TOP_C",
                "Temp_PY_BOT_C",
                "Temp_MIS7_C",
                "PD_Voltage_V",
            ]
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            print(f"IV DataFrame構築エラー: {e}")
            return pd.DataFrame()

    def analyze_data_structure(self) -> None:
        """入力HEX全体のサイズ感や64B分割の概観を表示（デバッグ）"""
        if not self.raw_data:
            print("データが読み込まれていません。")
            return
        all_hex = " ".join(self.raw_data).replace(" ", "")
        total_nibbles = len(all_hex)
        total_bytes = total_nibbles // 2
        print("=" * 60)
        print("データ構造解析")
        print("=" * 60)
        print(f"総16進文字数: {total_nibbles}")
        print(f"総バイト数: {total_bytes}")
        print(f"64B分割数: {total_bytes // 64}")
        print(f"余りバイト: {total_bytes % 64}")

    def export_to_binary(self, output_filename: str = "extracted_data.bin") -> None:
        """有効/無効に関わらず64Bパケットを連結してバイナリ保存（CRCはログのみ）"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_dir / output_filename
            if not Path(output_filename).is_absolute()
            else Path(output_filename)
        )

        count = 0
        with open(output_path, "wb") as fp:
            for packet_line in self.raw_data:
                hex_pairs = re.findall(r"[0-9A-Fa-f]{2}", packet_line)
                if len(hex_pairs) != self.PACKET_SIZE:
                    continue
                packet_bytes = [int(h, 16) for h in hex_pairs]

                if self.check_crc:
                    payload = packet_bytes[:-1]
                    received_crc = packet_bytes[-1]
                    calculated_crc = self.calc_crc8(payload)
                    if received_crc != calculated_crc:
                        print("  警告: バイナリ出力中にCRC不一致を検出 - 継続")
                fp.write(bytes(packet_bytes))
                count += 1
        print(f"{count}個のパケットを書き出し: '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIGS IV Data Packet Analysis Tool")
    parser.add_argument(
        "--mode",
        choices=["iv", "env", "piclog"],
        default=None,
        help="解析モード: iv=IV測定, env=環境データ, piclog=PICログCSV",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="入力ファイルのパス（未指定ならモード別既定ファイル: iv→iv_data.txt, env→env_data.txt, piclog→piclog_data.txt）",
    )
    args = parser.parse_args()

    # --mode が未指定なら対話的に選択してもらう
    mode_input = args.mode
    if mode_input is None:
        while True:
            try:
                mode_input = input("どのデータを解析しますか? [iv/env/piclog]: ").strip().lower()
            except EOFError:
                # 非対話環境で標準入力が閉じている場合は iv を既定値に
                mode_input = "iv"
            if mode_input in ("iv", "env", "piclog"):
                break
            print("入力が不正です。iv, env, piclog のいずれかを入力してください。")

    mode_map = {"iv": "iv_measurement", "env": "environment", "piclog": "piclog"}
    analysis_mode = mode_map[mode_input]

    # 入力ファイルの決定
    if args.file:
        input_file = args.file
    else:
        default_file_map = {
            "iv_measurement": "iv_data.txt",
            "environment": "env_data.txt",
            "piclog": "piclog_data.txt",
        }
        input_file = default_file_map[analysis_mode]

    analyzer = PacketAnalyzer(filename=input_file, analysis_mode=analysis_mode)
    data = analyzer.read_raw_data()
    if not data:
        return

    analyzer.analyze_packets()

    if analysis_mode == "environment":
        analyzer.export_environment_data_csv()
        return
    elif analysis_mode == "piclog":
        analyzer.export_packet_log_csv()
        return

    # IV測定データ: 工学値変換とJSON/グラフ（src配下を優先）
    try:
        import sys

        src_path = Path(__file__).parent / "src"
        # 先頭に挿入してトップレベルの同名ファイルより先に解決
        sys.path.insert(0, str(src_path))

        from engineering_converter import IVDataConverter  # type: ignore

        print("\n" + "=" * 80)
        print("工学値変換とグラフ出力...")
        print("=" * 80)

        converter = IVDataConverter(analysis_mode="iv_measurement")
        if converter.load_data():
            voltage_config = {"gain": 1.0, "offset": 0.0}
            current_config = {"gain": 1.0, "offset": 0.0, "shunt_r": 0.1}
            temp_config = {"method": "lmt84", "min": -50, "max": 150}

            print("\n工学値変換を実行中...")
            eng = converter.convert_to_engineering_units(
                voltage_config, current_config, temp_config
            )
            if eng is not None:
                converter.save_engineering_data_json(eng)
                converter.plot_iv_curve(eng)
                print("\n工学値変換完了")
            else:
                print("工学値変換に失敗しました")
        else:
            print("変換用データの読み込みに失敗しました")
    except ImportError as e:
        print(f"工学値変換モジュールのインポートエラー: {e}")
        print("src/engineering_converter.py が優先されるようパス設定を確認してください")
    except Exception as e:
        print(f"工学値変換エラー: {e}")


if __name__ == "__main__":
    main()
