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
4. 「記録終了」ボタンをクリック
5. 必要な数だけエピソードを繰り返し記録
6. Ctrl+C でOperatorを停止すると自動的にデータセット保存

保存されるデータ:
```
datasets/toio_dataset/
├── data.npz              # 全フレームデータ
└── meta/
    ├── info.json         # データセット情報
    └── episodes.json     # エピソード情報
```

### 2. データの再生

記録したエピソードを再生して確認：

```bash
# 最初のエピソードを再生
uv run python scripts/replay_dataset.py ./datasets/toio_dataset --episode 0

# 別のエピソードを再生
uv run python scripts/replay_dataset.py ./datasets/toio_dataset --episode 1
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
  "observation.state": [[x, y, collision], ...],  # shape: (N, 3)
    # x: ジョイスティックX軸 (-1.0 〜 1.0)
    # y: ジョイスティックY軸 (-1.0 〜 1.0)
    # collision: 衝突検知 (0.0 または 1.0)

  "action": [[left_motor, right_motor], ...],     # shape: (N, 2)
    # left_motor: 左モーター指令値 (-100 〜 100)
    # right_motor: 右モーター指令値 (-100 〜 100)

  "episode_index": [0, 0, ..., 1, 1, ...],        # shape: (N,)
  "frame_index": [0, 1, 2, ..., 0, 1, ...],       # shape: (N,)
  "timestamp": [0.0, 0.016, 0.033, ...],          # shape: (N,)
  "next.done": [False, ..., True, ...],           # shape: (N,)
}
```

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
│   ├── replay_dataset.py        # エピソード再生
│   ├── train.py                 # モデル訓練
│   └── inference.py             # 自律制御実行
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

## ライセンス

MIT

## 謝辞

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face のロボット学習フレームワーク
- [toio](https://toio.io/) - Sony の小型教育ロボット
