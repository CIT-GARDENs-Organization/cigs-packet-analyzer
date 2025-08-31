# JSON 要素リファレンス（ディレクトリ表記 + 表）

このドキュメントは、出力JSONの構造を「ディレクトリ（ツリー）表記」と「表形式」の2通りで示します。

## 1) ディレクトリ（ツリー）表記

```
root
├─ metadata (object)
│  ├─ conversion_timestamp: string
│  ├─ total_records: number
│  ├─ iv_continuation_packets: number
│  ├─ measurement_samples: number
│  └─ converter_version: string
├─ measurement_samples (array)
│  ├─ [i] (object)
│  │  ├─ sample_id: number
│  │  ├─ timestamp: number
│  │  ├─ environment (object)
│  │  │  ├─ pd_voltage_v: number
│  │  │  ├─ temp_py_top_c: number
│  │  │  ├─ temp_py_bot_c: number
│  │  │  └─ temp_mis7_c: number
│  │  └─ iv_measurements (array)
│  │     ├─ [j] (object)
│  │     │  ├─ step: number
│  │     │  └─ engineering_values (object)
│  │     │     ├─ voltage_v: number
│  │     │     ├─ current_a: number
│  │     │     └─ power_w: number
└─ packets (array, optional)
	├─ [k] (object)
	│  ├─ index: number
	│  ├─ crc (object)
	│  │  └─ valid: boolean
	│  ├─ hex: string
	│  └─ sample_number: number | null
```

---

## 2) 表での説明（概要）

### トップレベル
| キー | 型 | 説明 |
|---|---|---|
| metadata | オブジェクト | 変換メタ情報 |
| measurement_samples | 配列 | サンプルごとの測定データ |
| packets | 配列 | パケットのHEX/CRCサマリ（原文がある場合のみ） |

### metadata
| キー | 型 | 説明 |
|---|---|---|
| conversion_timestamp | 文字列(ISO8601) | 変換日時 |
| total_records | 数値 | 変換対象の総レコード数 |
| iv_continuation_packets | 数値 | 継続パケット数 |
| measurement_samples | 数値 | 測定サンプル数 |
| converter_version | 文字列 | 変換器バージョン |

備考: sensor_info, data_structure は出力しません。

### measurement_samples の各要素
| キー | 型 | 説明 |
|---|---|---|
| sample_id | 数値 | サンプルID |
| timestamp | 数値 | タイムスタンプ（FF由来） |
| environment | オブジェクト | 環境値（工学値のみ） |
| iv_measurements | 配列 | I-V測定の各ステップ |

#### environment
| キー | 型 | 説明 |
|---|---|---|
| pd_voltage_v | 数値 | PD電圧 [V] |
| temp_py_top_c | 数値 | 温度PY_TOP [°C] |
| temp_py_bot_c | 数値 | 温度PY_BOT [°C] |
| temp_mis7_c | 数値 | 温度MIS7 [°C] |

#### iv_measurements の各要素
| キー | 型 | 説明 |
|---|---|---|
| step | 数値 | ステップ番号 |
| engineering_values | オブジェクト | 電圧/電流/電力 |

##### engineering_values
| キー | 型 | 説明 |
|---|---|---|
| voltage_v | 数値 | 電圧 [V] |
| current_a | 数値 | 電流 [A] |
| power_w | 数値 | 電力 [W] |

### packets の各要素（任意）
| キー | 型 | 説明 |
|---|---|---|
| index | 数値 | パケット番号（1始まり） |
| crc | オブジェクト | CRC情報（有効性のみ） |
| hex | 文字列 | 64バイトHEX（空白区切り） |
| sample_number | 数値/null | 関連サンプル番号 |

#### crc
| キー | 型 | 説明 |
|---|---|---|
| valid | 真偽値 | CRC一致可否 |

---

補足:
- raw_data は出力しません（簡素化のため）。
- 上記以外のキーは含まれません。
