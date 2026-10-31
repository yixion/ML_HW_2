# 類神經網路自走車模擬器

本專案使用 NumPy 從零實作多層感知器（MLP），根據車輛感測器資料預測方向盤角度，並透過 Tkinter 顯示自走車模擬與軌跡重播。

專案支援兩種模型：

- 4D 模型：使用前方、右方、左方感測器距離預測方向盤角度。
- 6D 模型：額外加入車輛的 `x`、`y` 位置。

## 功能

- 載入 4D 與 6D 訓練資料
- 切分 training／validation dataset
- 特徵標準化
- 使用 NumPy 實作 MLP
- Mini-batch gradient descent
- 模型訓練、預測、儲存與載入
- 車輛運動模擬
- 前方、右方、左方距離感測器
- 邊界碰撞與終點判定
- 4D／6D 軌跡儲存與載入
- Tkinter GUI 動畫重播
- 車輛軌跡、感測器線與即時資料顯示
- 4D／6D replay 模式切換
- Pytest 自動化測試

## 執行環境

本專案已在以下環境測試：

- Ubuntu on WSL2
    - "Ubuntu 26.04 LTS"
- Python 3.14.4
- NumPy 2.5.4
- Pytest 9.1.1
- Tkinter

## 安裝

在專案根目錄建立虛擬環境：

```bash
python -m venv .venv
```

啟用虛擬環境：

```bash
source .venv/bin/activate
```

安裝相依套件：

```bash
python -m pip install numpy
python -m pip install pytest
```

如果 Ubuntu／WSL 缺少 Tkinter：

```bash
sudo apt update
sudo apt install python3-tk
```

## 執行程式

請在專案根目錄執行：

```bash
PYTHONPATH=src python main.py
```

程式會：

1. 載入賽道與訓練資料。
2. 載入已儲存的 4D、6D 模型。
3. 如果模型不存在，則先訓練並儲存模型。
4. 執行自走車模擬。
5. 儲存成功抵達終點的軌跡。
6. 開啟 Tkinter GUI。
7. 提供 4D 與 6D 模式的軌跡重播。

執行測式

```bash
PYTHONPATH=src python -m pytest -v
```

## GUI 操作

GUI 下方提供兩個按鈕：

- `4D model replay`
- `6D model replay`

按下按鈕後會從第一幀開始播放對應模型的模擬結果。

畫面包含：

- 賽道邊界
- 終點區域
- 車輛位置
- 車輛行駛軌跡
- 前方感測器
- 右方感測器
- 左方感測器
- 車輛位置與方向
- 感測器距離
- 目前 replay 模式與 frame

## 執行測試

執行全部測試：

```bash
PYTHONPATH=src python -m pytest -v
```

執行指定測試檔：

```bash
PYTHONPATH=src python -m pytest tests/test_simulator.py -v
```

## 專案結構

```text
ML_HW_2/
├── data/
│   ├── track.txt
│   ├── train4dAll.txt
│   └── train6dAll.txt
├── models/
│   ├── mlp_4d.npz
│   └── mlp_6d.npz
├── outputs/
│   ├── track4D.txt
│   └── track6D.txt
├── src/
│   └── autocar/
│       ├── gui/
│       │   └── app.py
│       ├── ml/
│       │   ├── dataset.py
│       │   ├── mlp.py
│       │   ├── model_io.py
│       │   ├── preprocessing.py
│       │   └── training.py
│       ├── simulation/
│       │   ├── collision.py
│       │   ├── geometry.py
│       │   ├── replay.py
│       │   ├── sensors.py
│       │   ├── simulator.py
│       │   ├── track.py
│       │   └── vehicle.py
│       └── storage/
│           └── trajectory.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## 模組說明

| 模組 | 功能 |
|---|---|
| `dataset.py` | 載入資料、分離 features／target、切分 training／validation |
| `preprocessing.py` | `StandardScaler` 的訓練、轉換與反轉換 |
| `mlp.py` | MLP 初始化、forward、backward、fit 與 predict |
| `training.py` | 4D、6D 模型的完整訓練流程 |
| `model_io.py` | 儲存與載入模型及 scaler |
| `track.py` | 賽道資料結構與賽道檔案載入 |
| `vehicle.py` | 車輛位置、方向與運動公式 |
| `geometry.py` | 射線交點與點到線段距離 |
| `sensors.py` | 計算前方、右方、左方感測器距離 |
| `collision.py` | 車輛碰撞與抵達終點判定 |
| `simulator.py` | 整合模型、感測器與車輛模擬 |
| `trajectory.py` | 4D、6D 軌跡儲存與載入 |
| `replay.py` | 將軌跡資料轉換為 GUI replay frames |
| `app.py` | Tkinter GUI 與動畫播放 |

## 資料格式

### 4D 訓練資料與軌跡

每一列包含：

```text
front_distance right_distance left_distance steering_degrees
```

### 6D 訓練資料與軌跡

每一列包含：

```text
x y front_distance right_distance left_distance steering_degrees
```

資料以空格分隔。

## 座標與角度

本專案在資料、模型輸出與物件狀態中使用 degree：

```text
heading 0°   → 向右
heading 90°  → 向上
heading 180° → 向左
heading 270° → 向下
```

方向盤角度限制為：

```text
-40° ≤ steering ≤ 40°
```

感測器方向：

```text
front = heading
right = heading - 45°
left  = heading + 45°
```

進入 `sin()`、`cos()` 等 NumPy 三角函式前，才將 degree 轉換為 radian。

## 模擬資料流程

```text
Track + Vehicle
       ↓
measure sensor distances
       ↓
StandardScaler.transform()
       ↓
MLPRegressor.predict()
       ↓
steering angle
       ↓
Vehicle.step()
       ↓
collision / goal detection
       ↓
SimulationRecord
       ↓
trajectory file
       ↓
Tkinter replay
```

## 注意事項

- 請從專案根目錄執行程式。
- 必須設定 `PYTHONPATH=src`。
- `right_distance` 與 `left_distance` 的欄位順序不可交換。
- Tkinter Canvas 的 `y` 軸向下，因此 GUI 會在 `world_to_canvas` 轉換時反轉 `y` 軸。
- GUI 動畫使用 `root.after()`，不使用 `time.sleep()`，以避免視窗停止回應。
