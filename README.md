# LeRobot × toio Web Controller

toio Core Cubeをブラウザから直感的に操作し、行動データを記録して機械学習モデルを訓練できるシステムです。

## 特徴

- **モバイルファースト UI**: スマホで片手操作可能なD-Padコントローラー
- **リアルタイム制御**: WebSocket経由で低遅延な操作
- **データ記録**: LeRobot形式でエピソードを記録
- **学習・推論**: 収集したデータから行動をクローニング

## システム構成

```
Web UI (ブラウザ)
    ↓ WebSocket
WebSocketサーバー (Python/FastAPI)
    ↓ WebSocket
Operator (Python)
    ↓ BLE
toio Core Cube
```

## セットアップ

### 1. 依存関係のインストール

[uv](https://github.com/astral-sh/uv) を使用します：

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトのセットアップ
uv sync
source .venv/bin/activate
```

### 2. 設定ファイルの準備

```bash
cp config.example.yaml config.yaml
```

`config.yaml` を編集して環境に合わせて設定：

```yaml
# toio の MAC アドレス (自動検出する場合は null のまま)
robot:
  mac_address: null  # または "XX:XX:XX:XX:XX:XX"

# データ記録の設定
recording:
  enabled: true
  output_dir: "./datasets"
```

## 使い方

### 基本的な遠隔操作

#### 1. WebSocketサーバーの起動

```bash
uv run python scripts/run_server.py
```

ブラウザで http://localhost:8765 を開く

#### 2. Operatorの起動

```bash
uv run python scripts/run_operator.py
```

toioに自動接続され、Webコントローラーで操作できます。

### 操作方法

**D-Padボタン:**
- 上ボタン: 前進
- 下ボタン: 後退
- 左ボタン: 左旋回
- 右ボタン: 右旋回

**キーボード (PCの場合):**
- W / ↑: 前進
- S / ↓: 後退
- A / ←: 左旋回
- D / →: 右旋回

**その他:**
- 速度スライダー: 速度を20%〜100%に調整
- 緊急停止ボタン: toioを即座に停止

## データ収集と学習

### 1. エピソードの記録

#### 記録開始の設定

`config.yaml` で記録を有効化：

```yaml
recording:
  enabled: true
  output_dir: "./datasets"
```

#### 記録の実行

1. WebSocketサーバーとOperatorを起動
2. Webコントローラーで「記録開始」ボタンをクリック
3. toioを操作してデモンストレーション
4. 「記録終了」ボタンをクリック（エピソードが自動保存されます）
5. 必要な数だけエピソードを繰り返し記録

保存されるデータ:
```
datasets/toio_dataset/
├── data.npz              # 全フレームデータ
└── meta/
    ├── info.json         # データセット情報
    └── episodes.json     # エピソード情報
```

### 2. データの確認 (オプション)

記録したデータセットを確認:

```bash
# データセット情報を表示
uv run python scripts/replay_dataset.py ./datasets/toio_dataset

# 特定のエピソードを再生
uv run python scripts/replay_dataset.py ./datasets/toio_dataset --episode 0
```

### 3. モデルの訓練

収集したデータから行動クローニングモデルを訓練：

```bash
uv run python scripts/train.py ./datasets/toio_dataset --epochs 100 --output ./models/policy.pth
```

オプション:
- `--epochs`: 訓練エポック数 (デフォルト: 100)
- `--batch-size`: バッチサイズ (デフォルト: 32)
- `--learning-rate`: 学習率 (デフォルト: 0.001)

訓練が完了すると、`./models/policy.pth` にモデルが保存されます。

### 4. 自律制御の実行

訓練したモデルでtoioを自律制御：

```bash
uv run python scripts/inference.py ./models/policy.pth
```

toioは訓練データから学習した行動パターンに従って自律的に動作します。
Ctrl+Cで停止できます。

## データフォーマット

### LeRobot互換データ構造

```python
{
  "observation.state": [[collision], ...],  # shape: (N, 1)
    # collision: 衝突検知フラグ (0.0 = 正常, 1.0 = 衝突)

  "action": [[left_motor, right_motor], ...],     # shape: (N, 2)
    # left_motor: 左モーター指令値 (-100 〜 100)
    # right_motor: 右モーター指令値 (-100 〜 100)

  "episode_index": [0, 0, ..., 1, 1, ...],        # shape: (N,)
  "frame_index": [0, 1, 2, ..., 0, 1, ...],       # shape: (N,)
  "timestamp": [0.0, 0.016, 0.033, ...],          # shape: (N,)
  "next.done": [False, ..., True, ...],           # shape: (N,)
}
```

### 観測空間の設計

**collision (衝突フラグ)**
- 通常時: 0.0
- 衝突検出時: 1.0
- toioの衝突センサーから取得

### 多様な回避行動の学習について

観測は衝突フラグのみですが、モデルは**データ収集時のユーザーの多様な回避行動**から学習します:
- 右回転、左回転、後退など、様々な回避パターンをデモンストレーション
- モデルは衝突時の状況(慣性、姿勢、ノイズなど)から暗黙的に多様な行動を学習
- 推論時、モデルの確率的な性質により、異なる回避行動が自然に生成されます

**重要**: 多様な回避行動を実現するには、データ収集時に意識的に様々なパターンで回避することが必要です

## プロジェクト構成

```
lerobot-toio-webctrl/
├── README.md
├── pyproject.toml
├── config.yaml                  # 設定ファイル
├── config.example.yaml
├── datasets/                    # 記録したデータセット
│   └── toio_dataset/
├── models/                      # 訓練済みモデル
│   └── policy.pth
├── scripts/                     # ユーティリティスクリプト
│   ├── run_server.py            # WebSocketサーバー起動
│   ├── run_operator.py          # オペレーター起動
│   ├── train.py                 # モデル訓練
│   ├── inference.py             # 自律制御実行
│   └── replay_dataset.py        # データセット再生(デバッグ用)
├── server/                      # WebSocketサーバー
│   ├── main.py
│   └── static/
│       ├── index.html           # Webコントローラー UI
│       ├── controller.js        # コントローラーロジック
│       └── styles.css
└── lerobot_operator/            # toio制御システム
    ├── __init__.py
    ├── run_operator.py          # メインループ
    ├── websocket_leader.py      # WebSocket通信
    ├── toio_driver.py           # toio BLE制御
    ├── mixing.py                # モーター制御ミキシング
    └── episode_recorder.py      # データ記録
```

## 高度な設定

### config.yaml の詳細

```yaml
ws:
  uri: "ws://127.0.0.1:8765/ws"
  ping_interval_sec: 1.0
  timeout_sec: 2.0

robot:
  mac_address: null               # toioのMACアドレス (自動検出: null)
  name_prefix: "toio Core Cube"
  scan_timeout_sec: 10.0
  scan_retry: 3

control:
  max_speed: 100                  # 最大速度 (0-100)
  deadzone: 0.08                  # デッドゾーン
  expo: 0.3                       # 指数カーブ
  slew_rate: 300                  # 加速度制限
  rate_hz: 60                     # 制御ループ周波数
  invert_x: false                 # X軸反転
  invert_y: false                 # Y軸反転

safety:
  estop_on_disconnect: true       # 切断時に緊急停止

recording:
  enabled: true                   # 記録機能の有効化
  output_dir: "./datasets"        # 出力ディレクトリ
```

## データ収集のベストプラクティス

### サンプリングレートの設定根拠

toioの技術仕様に基づいた推奨設定:

**制御ループ周波数 (rate_hz)**
- **推奨: 50-60Hz** (config.yamlのデフォルト: 60Hz)
- 根拠:
  - toio BLE接続間隔: 10-30ms (仕様)
  - toioモーター速度通知: 10Hz/100ms間隔
  - 60Hz = 16.7ms周期 → BLE接続間隔と整合
  - モーターコマンド持続時間: 50ms (3倍オーバーラップで連続性確保)

**データ記録の粒度**
- 観測(observation): 20-30Hz推奨
  - toioのセンサー更新頻度(10Hz)より高い
  - カメラなし構成では30Hzで十分
- 行動(action): 50-60Hz推奨
  - 制御ループと同期
  - 滑らかな動作の再現に必要

**エピソード設計**
- 長さ: 5-15秒/エピソード推奨
  - 短すぎると文脈が失われる
  - 長すぎると環境変化・ドリフトが混入
  - toioのような小型ロボットは10秒前後が最適
- 総数: 20-30エピソード以上
  - 多様な状況(直進、旋回、衝突回避)を含める
  - 衝突回避は複数パターン記録

### BLEの安定性

**通信安定化のヒント**
- Write Without Response使用(実装済み)
- 過度な送信頻度で詰まる場合は50Hzへ下げる
- macOS 10.12などで通知遅延がある場合は注意

**環境依存の調整**
- BLE混雑時: `rate_hz: 50` または `rate_hz: 30` へ下げる
- 推論時の安定性優先: `rate_hz: 50`
- データ収集時の高精度: `rate_hz: 60`

## トラブルシューティング

### toioが見つからない

- toioの電源が入っているか確認
- Bluetoothがオンになっているか確認
- `config.yaml` で `mac_address` を明示的に指定

### モデルの精度が低い

- より多くのエピソードを収集 (推奨: 20-30エピソード以上)
- 多様なシナリオを記録（直進、旋回、衝突回避など）
- 衝突回避のデモを複数パターン記録する
- エポック数を増やす (`--epochs 200`)
- バッチサイズを調整 (`--batch-size 64`)

### BLE接続が不安定

- 制御周波数を下げる: `config.yaml` で `rate_hz: 50` または `30`
- 他のBluetooth機器との干渉を避ける
- toioとの距離を近づける(推奨: 2m以内)

## ライセンス

MIT

## 謝辞

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face のロボット学習フレームワーク
- [toio](https://toio.io/) - Sony の小型教育ロボット
