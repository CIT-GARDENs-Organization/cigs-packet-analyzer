#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIGS IV Data JSON Exporter
工学値変換後のデータをJSONファイルで出力する

入出力パス:
- 入力: 親フォルダ内 output/parsed_iv_data.csv（デフォルト）
- 出力: 親フォルダ内 output/ 配下に保存（自動作成）
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

class JSONExporter:
    def __init__(self):
        """
        JSON出力クラスの初期化
        """
        # ベースディレクトリ（このスクリプトの親フォルダ）
        self.base_dir = Path(__file__).resolve().parent.parent
        self.output_dir = self.base_dir / "output"
        
    def save_engineering_data(self, data: pd.DataFrame, output_filename: str = None, 
                            converter_config=None):
        """
        工学値変換後のデータをJSONファイルで保存
        FFパケットのヘッダー情報と継続パケットを適切に構造化
        
        Args:
            data (pd.DataFrame): 変換後データ
            output_filename (str): 出力ファイル名（Noneの場合は日時付きで自動生成）
            converter_config: 変換設定オブジェクト
        """
        try:
            # 出力先ディレクトリ作成
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル名に日時を追加
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                output_filename = f"iv_data_{timestamp}.json"
            
            # 拡張子をjsonに変更
            if output_filename.endswith('.csv'):
                output_filename = output_filename.replace('.csv', '.json')
            elif not output_filename.endswith('.json'):
                output_filename += '.json'
            
            output_path = (self.output_dir / output_filename) if not Path(output_filename).is_absolute() else Path(output_filename)
            
            # センサー情報は出力しない（簡潔化）
            
            # パケットタイプ別の統計情報
            ff_packets = data[data['PacketType'] == 'FF_Header'] if 'PacketType' in data.columns else pd.DataFrame()
            iv_packets = data[data['PacketType'] == 'IV_Continuation'] if 'PacketType' in data.columns else pd.DataFrame()
            
            # 測定サンプル情報の構築（SampleNumber列を使用）
            measurement_samples = []
            if 'SampleNumber' in data.columns:
                # SampleNumber別にサンプルを分割
                sample_numbers = data['SampleNumber'].unique()
                sample_numbers = sorted([s for s in sample_numbers if pd.notna(s) and s > 0])
                
                for sample_num in sample_numbers:
                    sample_data = data[data['SampleNumber'] == sample_num]
                    
                    if len(sample_data) > 0:
                        # サンプルの最初のデータ（FFパケット）から環境データを取得
                        ff_data = sample_data.iloc[0]
                        
                        sample_info = {
                            "sample_id": int(sample_num),
                            "timestamp": int(ff_data['Timestamp']) if pd.notna(ff_data['Timestamp']) else 0,
                            "environment": {
                                # 工学値のみ
                                "pd_voltage_v": float(ff_data['PD_Voltage_V']) if 'PD_Voltage_V' in ff_data and pd.notna(ff_data['PD_Voltage_V']) else 0.0,
                                "temp_py_top_c": float(ff_data['Temp_PY_TOP_C']) if 'Temp_PY_TOP_C' in ff_data and pd.notna(ff_data['Temp_PY_TOP_C']) else 0.0,
                                "temp_py_bot_c": float(ff_data['Temp_PY_BOT_C']) if 'Temp_PY_BOT_C' in ff_data and pd.notna(ff_data['Temp_PY_BOT_C']) else 0.0,
                                "temp_mis7_c": float(ff_data['Temp_MIS7_C']) if 'Temp_MIS7_C' in ff_data and pd.notna(ff_data['Temp_MIS7_C']) else 0.0
                            },
                            "iv_measurements": []
                        }
                        
                        # このサンプルのIV測定データを追加
                        for _, row in sample_data.iterrows():
                            measurement = {
                                "step": int(row['Step']) if pd.notna(row['Step']) else 0,
                                "engineering_values": {
                                    "voltage_v": float(row['Voltage_V']) if 'Voltage_V' in row and pd.notna(row['Voltage_V']) else 0.0,
                                    "current_a": float(row['Current_A']) if 'Current_A' in row and pd.notna(row['Current_A']) else 0.0,
                                    "power_w": float(row['Power_W']) if 'Power_W' in row and pd.notna(row['Power_W']) else 0.0
                                }
                            }
                            sample_info["iv_measurements"].append(measurement)
                        
                        measurement_samples.append(sample_info)
            
            # パケットサマリを構築（raw_data.txt から64バイトHEXとCRCを抽出）
            packets_summary = []
            try:
                raw_path = self.base_dir / 'raw_data.txt'
                if raw_path.exists():
                    with open(raw_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    hex_pairs = re.findall(r'[0-9A-Fa-f]{2}', content)
                    # 64バイトずつ
                    packets = []
                    for i in range(0, len(hex_pairs), 64):
                        pkt = hex_pairs[i:i+64]
                        if len(pkt) == 64:
                            packets.append(pkt)

                    # DataFrame からパケット→サンプル番号の対応（有効パケットのみ）
                    packet_sample_map = {}
                    if 'Packet' in data.columns and 'SampleNumber' in data.columns:
                        first_rows = data.sort_values('Step').drop_duplicates(subset=['Packet'])
                        for _, r in first_rows.iterrows():
                            packet_sample_map[int(r['Packet'])] = int(r['SampleNumber']) if pd.notna(r['SampleNumber']) else None

                    def xor_crc(byte_list):
                        if not byte_list:
                            return 0
                        c = byte_list[0]
                        for b in byte_list[1:]:
                            c ^= b
                        return c & 0xFF

                    for idx, pkt in enumerate(packets, start=1):
                        bytes_int = [int(h, 16) for h in pkt]
                        received = bytes_int[-1]
                        calculated = xor_crc(bytes_int[:-1])
                        packets_summary.append({
                            'index': idx,
                            'crc': {
                                'valid': bool(received == calculated)
                            },
                            'hex': ' '.join(pkt),
                            'sample_number': packet_sample_map.get(idx)
                        })
            except Exception:
                # パケットサマリは任意項目のため、失敗しても処理継続
                packets_summary = []

            # DataFrameをJSONに変換して保存
            json_data = {
                "metadata": {
                    "conversion_timestamp": datetime.now().isoformat(),
                    "total_records": len(data),
                    "iv_continuation_packets": len(iv_packets),
                    "measurement_samples": len(measurement_samples),
                    "converter_version": "1.0.0"
                },
                "measurement_samples": measurement_samples,
                "packets": packets_summary
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"工学値データを '{output_path}' に保存しました (JSON形式)")
            print(f"- 測定サンプル数: {len(measurement_samples)}")
            print(f"- FFヘッダーパケット: {len(ff_packets)}")
            print(f"- IV継続パケット: {len(iv_packets)}")
        except Exception as e:
            print(f"JSON保存エラー: {e}")

def main():
    """メイン実行関数 - テスト用"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.engineering_converter import IVDataConverter
    
    print("CIGS IV Data JSON Exporter")
    print("=" * 50)
    
    # データ変換クラス初期化
    converter = IVDataConverter()
    
    # データ読み込み
    if not converter.load_data():
        return
    
    # 工学値変換実行
    print("\n工学値変換を実行中...")
    engineering_data = converter.convert_to_engineering_units()
    
    if engineering_data is not None:
        # JSON出力
        exporter = JSONExporter()
        exporter.save_engineering_data(engineering_data, converter_config=converter.converter.config)
        
    print("\n変換完了！")
    print("- 工学値データ: output/iv_data_YYYYMMDD_HHMM.json")

if __name__ == "__main__":
    main()
