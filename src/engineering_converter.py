#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIGS IV Data Engineering Unit Converter
12bitのADCデータを工学値（電圧、電流、温度など）に変換する

12bitデータ範囲: 0 - 4095 (0x000 - 0xFFF)

入出力パス:
- 入力: 親フォルダ内 raw_data.txt（CSVは作成しない）
- 出力: 親フォルダ内 output/ 配下に保存（自動作成）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# JSON出力機能をインポート
try:
    from .json_exporter import JSONExporter
except ImportError:
    # 相対インポートが失敗した場合の代替
    import sys
    sys.path.append(str(Path(__file__).parent))
    from json_exporter import JSONExporter

@dataclass
class ConversionConfig:
    """変換設定クラス"""
    # ADC基準電圧 (V)
    vref: float = 2.5
    
    # ADC分解能 (12bit)
    adc_resolution: int = 12
    adc_max_value: int = 4095  # 2^12 - 1
    
    # 電圧値変換用の変換係数
    voltage_gain: float = 1.0      # 電圧ゲイン
    voltage_offset: float = 0.0    # 電圧オフセット (V)
    
    # 電流値変換用の変換係数
    current_gain: float = 200.0      # 電流ゲイン
    current_offset: float = 0.0    # 電流オフセット (A)
    current_shunt_r: float = 0.1   # シャント抵抗 (Ω)
    current_vref: float = 1.24     # 電流センサ基準電圧 (V)
    
    # 温度変換係数 (LMT84線形温度センサ)
    temp_a: float = 0.001129148    # Steinhart-Hart係数 A (サーミスタ用、参考値)
    temp_b: float = 0.000234125    # Steinhart-Hart係数 B (サーミスタ用、参考値)
    temp_c: float = 0.0000000876741  # Steinhart-Hart係数 C (サーミスタ用、参考値)
    temp_r_ref: float = 10000.0    # 基準抵抗値 (Ω) (サーミスタ用、参考値)
    temp_linear_min: float = -50.0 # LMT84動作温度範囲最小 (℃)
    temp_linear_max: float = 150.0 # LMT84動作温度範囲最大 (℃)
    # LMT84特性パラメータ
    lmt84_v0: float = 1.8639       # 0℃での出力電圧 (V)
    lmt84_tc: float = -0.0055      # 温度係数 (-5.5mV/℃)
    
    # 照度変換係数
    illuminance_gain: float = 1.0      # 照度ゲイン
    illuminance_offset: float = 0.0    # 照度オフセット (lux)
    illuminance_max: float = 10000.0   # 最大照度 (lux)

class EngineeringConverter:
    def __init__(self, config: ConversionConfig = None):
        """
        工学値変換クラスの初期化
        
        Args:
            config (ConversionConfig): 変換設定
        """
        self.config = config if config else ConversionConfig()
        
    def adc_to_voltage(self, adc_value: int) -> float:
        """
        12bit ADC値を電圧に変換
        
        Args:
            adc_value (int): ADC値 (0-4095)
            
        Returns:
            float: 電圧値 (V)
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')
        
        # ADC値を基準電圧に対する比率で電圧に変換
        voltage = (adc_value / self.config.adc_max_value) * self.config.vref
        return voltage * self.config.voltage_gain + self.config.voltage_offset
    
    def adc_to_current(self, adc_value: int, shunt_resistance: float = None) -> float:
        """
        12bit ADC値を電流に変換（シャント抵抗使用想定）
        
        Args:
            adc_value (int): ADC値 (0-4095)
            shunt_resistance (float): シャント抵抗値 (Ω) - Noneの場合は設定値を使用
            
        Returns:
            float: 電流値 (A)
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')
        
        # ADC値を電流センサ専用の基準電圧で電圧に変換
        voltage = (adc_value / self.config.adc_max_value) * self.config.vref
        
        # シャント抵抗値の決定
        if shunt_resistance is None:
            shunt_resistance = self.config.current_shunt_r
        
        # 電流計算: (V - current_vref) / current_shunt_r / current_gain
        current = (voltage - self.config.current_vref) / shunt_resistance / self.config.current_gain
        return current + self.config.current_offset
    
    def adc_to_temperature_steinhart_hart(self, adc_value: int) -> float:
        """
        12bit ADC値を温度に変換（Steinhart-Hart方程式使用）
        
        Args:
            adc_value (int): ADC値 (0-4095)
            
        Returns:
            float: 温度 (℃)
        """
        voltage = self.adc_to_voltage(adc_value)
        if np.isnan(voltage) or voltage <= 0:
            return float('nan')
        
        # 分圧回路から抵抗値計算
        if voltage >= self.config.vref:
            return float('nan')
        
        resistance = self.config.temp_r_ref * voltage / (self.config.vref - voltage)
        
        if resistance <= 0:
            return float('nan')
        
        # Steinhart-Hart方程式
        ln_r = np.log(resistance)
        temp_k = 1.0 / (self.config.temp_a + 
                       self.config.temp_b * ln_r + 
                       self.config.temp_c * ln_r**3)
        
        # ケルビンから摂氏に変換
        temp_c = temp_k - 273.15
        return temp_c
    
    def adc_to_temperature_lmt84(self, adc_value: int) -> float:
        """ユーザー提供の二次式（Excel）による温度換算を適用。

        =(5.506-SQRT(POWER(-5.506,2)+4*0.00176*(870.6-B3)))/(2*(-0.00176))+30
        B3: 電圧[mV]
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')

        # ADC -> 電圧[V] -> mV
        v_mV = (adc_value / self.config.adc_max_value) * self.config.vref * 1000.0
        a = 0.00176
        b = -5.506
        c = 870.6 - v_mV
        disc = b * b + 4 * a * c
        if disc < 0:
            disc = 0.0
        try:
            import math
            sqrt_disc = math.sqrt(disc)
            temp = (5.506 - sqrt_disc) / (2 * a) + 30.0
            return temp
        except Exception:
            return float('nan')
    
    def adc_to_temperature_linear(self, adc_value: int, 
                                temp_min: float = None, 
                                temp_max: float = None) -> float:
        """
        12bit ADC値を温度に変換（線形変換）
        
        Args:
            adc_value (int): ADC値 (0-4095)
            temp_min (float): 最小温度 (℃) - Noneの場合は設定値を使用
            temp_max (float): 最大温度 (℃) - Noneの場合は設定値を使用
            
        Returns:
            float: 温度 (℃)
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')
        
        # 温度範囲の決定
        if temp_min is None:
            temp_min = self.config.temp_linear_min
        if temp_max is None:
            temp_max = self.config.temp_linear_max
        
        # 線形変換
        temp_range = temp_max - temp_min
        temperature = temp_min + (adc_value / self.config.adc_max_value) * temp_range
        return temperature
    
    def adc_to_illuminance_linear(self, adc_value: int) -> float:
        """
        12bit ADC値を照度に変換（線形変換）
        
        Args:
            adc_value (int): ADC値 (0-4095)
            
        Returns:
            float: 照度 (lux)
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')
        
        # 線形変換（0 lux から最大照度まで）
        illuminance = (adc_value / self.config.adc_max_value) * self.config.illuminance_max
        return illuminance * self.config.illuminance_gain + self.config.illuminance_offset
    
    def adc_to_illuminance_logarithmic(self, adc_value: int) -> float:
        """
        12bit ADC値を照度に変換（対数変換 - フォトダイオード特性）
        
        Args:
            adc_value (int): ADC値 (0-4095)
            
        Returns:
            float: 照度 (lux)
        """
        if adc_value < 0 or adc_value > self.config.adc_max_value:
            return float('nan')
        
        if adc_value <= 0:
            return 0.0
        
        # 対数変換（フォトダイオードの特性を考慮）
        voltage = self.adc_to_voltage(adc_value)
        if voltage <= 0:
            return 0.0
        
        # 簡単な対数モデル: lux = 10^(voltage * scale) - 1
        # より正確にはセンサーの特性に応じて調整が必要
        scale = 3.0  # 調整可能パラメータ
        illuminance = (10 ** (voltage * scale)) - 1
        
        # 最大値制限
        illuminance = min(illuminance, self.config.illuminance_max)
        
        return illuminance * self.config.illuminance_gain + self.config.illuminance_offset

class IVDataConverter:
    def __init__(self, csv_filename: str = "output/parsed_iv_data.csv", analysis_mode: str = "iv_measurement"):
        """
        IV測定データ変換クラス
        
        Args:
            csv_filename (str): 互換用（未使用）。入力は raw_data.txt から直接解析します。
            analysis_mode (str): 解析モード ("iv_measurement" または "environment")
        """
        # ベースディレクトリ（このスクリプトの親フォルダ）
        self.base_dir = Path(__file__).resolve().parent.parent
        self.output_dir = self.base_dir / "output"
        self.csv_filename = str((self.base_dir / csv_filename)) if not Path(csv_filename).is_absolute() else csv_filename
        self.data = None
        self.converter = EngineeringConverter()
        self.analysis_mode = analysis_mode
        
    def load_data(self) -> bool:
        """
        raw_data.txt から解析し、DataFrameを生成して読み込み（CSVは作らない）
        
        Returns:
            bool: 読み込み成功可否
        """
        try:
            import sys
            from pathlib import Path
            base_dir = Path(__file__).resolve().parent.parent
            sys.path.append(str(base_dir))
            from packet_analysis import PacketAnalyzer

            # 入力ファイルは PacketAnalyzer に委ねる（モード別既定: iv→iv_data.txt, env→env_data.txt）
            analyzer = PacketAnalyzer(filename=None, analysis_mode=self.analysis_mode)
            raw_packets = analyzer.read_raw_data()
            if not raw_packets:
                print("エラー: 入力データの読み込みに失敗しました（ファイル未検出または空）")
                return False

            df = analyzer.build_parsed_dataframe()
            if df is None or df.empty:
                print("エラー: 解析データが空です")
                return False

            self.data = df
            mode_name = "環境データ" if self.analysis_mode == "environment" else "IV測定データ"
            print(f"データ読み込み完了: {len(self.data)}行 ({mode_name})")
            return True
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False
    
    def convert_to_engineering_units(self, 
                                   voltage_config: Dict = None,
                                   current_config: Dict = None,
                                   temp_config: Dict = None) -> pd.DataFrame:
        """
        工学値変換を実行
        
        Args:
            voltage_config (Dict): 電圧変換設定
            current_config (Dict): 電流変換設定  
            temp_config (Dict): 温度変換設定
            
        Returns:
            pd.DataFrame: 工学値変換後のデータ
        """
        if self.data is None:
            print("データが読み込まれていません")
            return None
        
        # デフォルト設定
        v_config = voltage_config or {'gain': 1.0, 'offset': 0.0}
        i_config = current_config or {'gain': 1.0, 'offset': 0.0, 'shunt_r': 0.1}
        t_config = temp_config or {'method': 'linear', 'min': -40, 'max': 125}
        
        # 変換設定を更新
        config = ConversionConfig()
        config.voltage_gain = v_config.get('gain', 1.0)
        config.voltage_offset = v_config.get('offset', 0.0)
        config.current_gain = i_config.get('gain', 1.0)
        config.current_offset = i_config.get('offset', 0.0)
        
        self.converter.config = config
        
        # 新しいデータフレームを作成
        result_data = self.data.copy()
        
        # Data0を電圧に変換 (仮定: Data0 = 電圧測定値)
        print("Data0を電圧に変換中...")
        result_data['Voltage_V'] = result_data['Data0'].apply(
            lambda x: self.converter.adc_to_voltage(x)
        )
        
        # Data1を電流に変換 (仮定: Data1 = 電流測定値)
        print("Data1を電流に変換中...")
        shunt_resistance = i_config.get('shunt_r', 0.1)
        result_data['Current_A'] = result_data['Data1'].apply(
            lambda x: self.converter.adc_to_current(x, shunt_resistance)
        )
        
        # 電力計算
        result_data['Power_W'] = result_data['Voltage_V'] * result_data['Current_A']
        
        # 温度データ変換
        temp_method = t_config.get('method', 'lmt84')  # デフォルトをLMT84に変更
        print(f"温度データを変換中... (方法: {temp_method})")
        
        if temp_method == 'lmt84':
            # LMT84温度センサ変換
            result_data['Temp_PY_TOP_C'] = result_data['Temp_PY_TOP'].apply(
                self.converter.adc_to_temperature_lmt84
            )
            result_data['Temp_PY_BOT_C'] = result_data['Temp_PY_BOT'].apply(
                self.converter.adc_to_temperature_lmt84
            )
            result_data['Temp_MIS7_C'] = result_data['Temp_MIS7'].apply(
                self.converter.adc_to_temperature_lmt84
            )
        elif temp_method == 'linear':
            temp_min = t_config.get('min', -50)
            temp_max = t_config.get('max', 150)
            result_data['Temp_PY_TOP_C'] = result_data['Temp_PY_TOP'].apply(
                lambda x: self.converter.adc_to_temperature_linear(x, temp_min, temp_max)
            )
            result_data['Temp_PY_BOT_C'] = result_data['Temp_PY_BOT'].apply(
                lambda x: self.converter.adc_to_temperature_linear(x, temp_min, temp_max)
            )
            result_data['Temp_MIS7_C'] = result_data['Temp_MIS7'].apply(
                lambda x: self.converter.adc_to_temperature_linear(x, temp_min, temp_max)
            )
        else:  # Steinhart-Hart (サーミスタ用、参考として残す)
            result_data['Temp_PY_TOP_C'] = result_data['Temp_PY_TOP'].apply(
                self.converter.adc_to_temperature_steinhart_hart
            )
            result_data['Temp_PY_BOT_C'] = result_data['Temp_PY_BOT'].apply(
                self.converter.adc_to_temperature_steinhart_hart
            )
            result_data['Temp_MIS7_C'] = result_data['Temp_MIS7'].apply(
                self.converter.adc_to_temperature_steinhart_hart
            )
        
        # PD値も電圧に変換
        result_data['PD_Voltage_V'] = result_data['PD'].apply(
            lambda x: self.converter.adc_to_voltage(x)
        )
        
        return result_data
    
    def save_engineering_data_json(self, data: pd.DataFrame, output_filename: str = None):
        """
        工学値変換後のデータをJSONファイルで保存
        
        Args:
            data (pd.DataFrame): 変換後データ
            output_filename (str): 出力ファイル名（Noneの場合は日時付きで自動生成）
        """
        try:
            exporter = JSONExporter()
            exporter.save_engineering_data(data, output_filename, self.converter.config)
        except Exception as e:
            print(f"JSON保存エラー: {e}")
    
    def plot_iv_curve(self, data: pd.DataFrame):
        """
        I-V特性グラフを作成・表示
        FFパケットで区切られた測定セッション別に色分けして表示
        
        Args:
            data (pd.DataFrame): 工学値変換後データ
        """
        try:
            # Use default matplotlib font (no explicit Japanese font)

            # Create figure
            plt.figure(figsize=(12, 8))

            # If SampleNumber exists, plot per sample
            if 'SampleNumber' in data.columns:
                sample_numbers = data['SampleNumber'].unique()
                sample_numbers = sorted([s for s in sample_numbers if pd.notna(s) and s > 0])

                if sample_numbers:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_numbers)))

                    for i, sample_num in enumerate(sample_numbers):
                        sample_data = data[data['SampleNumber'] == sample_num]

                        if len(sample_data) > 0:
                            ff_data = sample_data.iloc[0]
                            timestamp = int(ff_data['Timestamp']) if pd.notna(ff_data['Timestamp']) else 0
                            temp_avg = 0
                            if all(col in ff_data for col in ['Temp_PY_TOP_C', 'Temp_PY_BOT_C', 'Temp_MIS7_C']):
                                temps = [ff_data['Temp_PY_TOP_C'], ff_data['Temp_PY_BOT_C'], ff_data['Temp_MIS7_C']]
                                valid_temps = [t for t in temps if pd.notna(t)]
                                temp_avg = np.mean(valid_temps) if valid_temps else 0

                            voltage = sample_data['Voltage_V'].values
                            current = sample_data['Current_A'].values * 1000  # mA

                            label = f'Sample {int(sample_num)} (T=0x{timestamp:08X}, Temp={temp_avg:.1f} C)'

                            plt.plot(
                                voltage,
                                current,
                                'o-',
                                color=colors[i],
                                label=label,
                                markersize=3,
                                linewidth=1.5,
                                alpha=0.8,
                            )
                else:
                    # No samples -> plot all as one
                    voltage = data['Voltage_V'].values
                    current = data['Current_A'].values * 1000
                    plt.plot(
                        voltage,
                        current,
                        'bo-',
                        markersize=3,
                        linewidth=1.5,
                        alpha=0.8,
                        label='All data',
                    )
            elif 'Packet' in data.columns:
                packets = data['Packet'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(packets)))

                for i, packet_num in enumerate(packets):
                    packet_data = data[data['Packet'] == packet_num]
                    v = packet_data['Voltage_V'].values
                    i_ma = packet_data['Current_A'].values * 1000
                    plt.plot(
                        v,
                        i_ma,
                        'o-',
                        color=colors[i],
                        label=f'Packet {packet_num}',
                        markersize=3,
                        linewidth=1.5,
                        alpha=0.8,
                    )
            else:
                voltage = data['Voltage_V'].values
                current = data['Current_A'].values * 1000
                plt.plot(
                    voltage,
                    current,
                    'bo-',
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.8,
                )

            # Labels and title in English
            plt.xlabel('Voltage [V]', fontsize=12)
            plt.ylabel('Current [mA]', fontsize=12)
            plt.title('CIGS Solar Cell I-V Curve\n(by sample)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # Legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Layout
            plt.tight_layout()

            # Show
            plt.show()

            print('Plotted I-V curves (by sample)')

        except Exception as e:
            print(f'Plot error: {e}')
            print('matplotlib might not be installed.')
            print('pip install matplotlib')

def main():
    """メイン実行関数"""
    print("CIGS IV Data Engineering Unit Converter")
    print("=" * 50)
    
    # 解析モードの選択（デフォルトはIV測定）
    analysis_mode = "iv_measurement"  # または "environment"
    
    # データ変換クラス初期化
    converter = IVDataConverter(analysis_mode=analysis_mode)
    
    # データ読み込み
    if not converter.load_data():
        return
    
    if analysis_mode == "environment":
        # 環境データの場合は工学値変換のみ実行
        print("\n環境データ解析モード")
        
        # 環境データの工学値変換
        engineering_data = converter.data.copy()
        
        # PD値を電圧に変換（8bit値を12bitスケールに変換してから）
        engineering_data['PD_Voltage_V'] = engineering_data['PD'].apply(
            lambda x: converter.converter.adc_to_voltage(x * 16)  # 8bit→12bit変換
        )
        
        # 温度データをLMT84で変換（12bit値）
        engineering_data['Temp_PY_TOP_C'] = engineering_data['Temp_PY_TOP'].apply(
            converter.converter.adc_to_temperature_lmt84
        )
        engineering_data['Temp_PY_BOT_C'] = engineering_data['Temp_PY_BOT'].apply(
            converter.converter.adc_to_temperature_lmt84
        )
        engineering_data['Temp_MIS7_C'] = engineering_data['Temp_MIS7'].apply(
            converter.converter.adc_to_temperature_lmt84
        )
        
        # JSONファイルで保存（環境データ用の構造）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        converter.save_engineering_data_json(engineering_data, f"environment_data_{timestamp}.json")
        
        print("\n環境データ変換完了！")
        print(f"- 環境データCSV: output/environment_data.csv")
        print(f"- 環境データJSON: output/environment_data_{timestamp}.json")
        
    else:
        # IV測定データの場合は従来通りの処理
        print("\nIV測定データ解析モード")
        
        # 変換設定の例（実際の測定系に合わせて調整してください）
        voltage_config = {
            'gain': 1.0,        # 電圧ゲイン
            'offset': 0.0       # 電圧オフセット (V)
        }
        
        current_config = {
            'gain': 1.0,        # 電流ゲイン  
            'offset': 0.0,      # 電流オフセット (A)
            'shunt_r': 0.1      # シャント抵抗 (Ω)
        }
        
        temp_config = {
            'method': 'lmt84',  # 'lmt84', 'linear', または 'steinhart_hart'
            'min': -50,         # 最小温度 (℃) - LMT84範囲
            'max': 150          # 最大温度 (℃) - LMT84範囲
        }
        
        # 工学値変換実行
        print("\n工学値変換を実行中...")
        engineering_data = converter.convert_to_engineering_units(
            voltage_config, current_config, temp_config
        )
        
        if engineering_data is not None:
            # JSONファイルで保存
            converter.save_engineering_data_json(engineering_data)
            
            # I-V特性グラフを表示
            converter.plot_iv_curve(engineering_data)
            
            print("\n変換完了！")
            print("- 工学値データ: output/iv_data_YYYYMMDD_HHMM.json")
            print("- I-V特性グラフを表示しました")

if __name__ == "__main__":
    main()
