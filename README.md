# LeRobot × toio Web Controller

toio Core Cubeの**衝突センサーを活用した自律的な障害物回避行動**を、行動クローニング（Behavior Cloning）で学習するシステムです。

Webブラウザからtoioを操作してデモンストレーションを記録し、機械学習モデルが衝突検知からの回避行動を学習します。学習後のモデルはtoioを自律制御し、壁や障害物にぶつかると自動的に回避行動を実行します。

## 主な特徴

- **衝突検知による回避学習**: toioの衝突センサーから回避行動を学習
  - ユーザーが様々な回避パターン（右回転、左回転、後退など）をデモンストレーション
  - モデルが衝突時の多様な回避行動を学習
  - Roombaのような自律走行を実現

- **純粋な行動クローニング**: ユーザーのデモンストレーションのみから学習
  - 観測: 衝突フラグ（0.0 or 1.0）のみのシンプルな入力
  - 行動: 左右モーター速度（-100〜100）
  - 合成データやルールベースのロジックなし

- **Web UI による直感的なデータ収集**:
  - スマホ/PCブラウザから操作
  - リアルタイムなテレオペレーション
  - ワンクリックでエピソード記録

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

このシステムの核心は、**衝突フラグという最小限の情報から回避行動を学習する**点です。

**collision (衝突フラグ)**
- 通常時: 0.0
- 衝突検出時: 1.0
- toioの衝突センサーから直接取得

### 衝突検知から回避行動を学習する仕組み

**1. データ収集フェーズ**
- ユーザーがWebコントローラーでtoioを操作
- 壁にぶつかったら、様々な方法で回避をデモンストレーション:
  - 右回転して回避
  - 左回転して回避
  - 後退してから方向転換
  - など、多様なパターンを実演
- 衝突センサーの状態とモーター指令が記録される

**2. 学習フェーズ**
- モデルは `collision=0.0 → 前進` のパターンを学習
- モデルは `collision=1.0 → 様々な回避行動` のパターンを学習
- 同じ衝突状態でも、デモンストレーションの多様性から異なる回避パターンを学習

**3. 推論フェーズ**
- toioが自律走行中に壁にぶつかる
- 衝突センサーが反応（collision=1.0）
- モデルが学習した回避行動を出力
- モデルの確率的性質により、実行のたびに異なる回避パターンが選ばれる

**重要なポイント**:
- 観測は衝突フラグのみだが、多様な回避行動が実現できる
- 鍵はデータ収集時に**意識的に様々なパターンで回避をデモンストレーション**すること
- 合成データや人工的なパターン生成は一切不要
- 純粋にユーザーの操作から学習

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
  deadzone: 0.08                  # デッドゾーン（微小入力を無視）
  expo: 0.0                       # 指数カーブ (0.0=線形, 学習用推奨)
  slew_rate: 600                  # 加速度制限 (大きいほど即応性高い)
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

### コントローラー設定の調整

**学習に適した設定（推奨）**:
- `expo: 0.0` - 線形応答で明確な入力信号
- `slew_rate: 600` - 高い応答性で意図が明確
- Web UIの`turningSensitivity: 0.7` - 左右旋回が明確に

**メリット**:
- ユーザーの意図（前進/後退/左右旋回）が明確にデータに反映
- モデルが学習しやすい明瞭なパターン
- でも完全なバイナリ入力ではないため、自然な動作

**滑らかな操作感が欲しい場合**:
- `expo: 0.2-0.3` - 指数カーブで微妙な制御が可能
- `slew_rate: 200-300` - よりスムーズな加減速
- ただし学習データとしては曖昧になるため、訓練後の推論用に使用推奨

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
- 観測(observation)と行動(action): 制御ループと同期（デフォルト60Hz）
  - 実装では`fps=rate_hz`で制御周波数と記録周波数が一致
  - toioのセンサー更新頻度(10Hz)より十分高い
  - 滑らかな動作の再現に必要な時間解像度を確保

**エピソード設計**
- 長さ: 5-15秒/エピソード推奨
  - 短すぎると文脈が失われる
  - 長すぎると環境変化・ドリフトが混入
  - toioのような小型ロボットは10秒前後が最適
- 総数: 20-30エピソード以上
  - **特に重要**: 衝突回避パターンを意識的に多様化
    - 右回転での回避を数回
    - 左回転での回避を数回
    - 後退してから方向転換を数回
    - 急旋回での回避を数回
  - 通常の走行も含める（壁のない直進、旋回など）

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

### 衝突時に同じ回避行動しかしない（回避パターンが単調）

**原因**: データ収集時の回避パターンが偏っている

**解決策**:
- データ収集を見直し、意識的に多様な回避パターンをデモンストレーション:
  - 右回転での回避
  - 左回転での回避
  - 後退してから旋回
  - 急旋回、緩旋回など
- 各パターンを複数回記録して、データセットのバランスを取る
- 最低でも3-4種類の異なる回避パターンを含めることを推奨

### モデルの精度が低い・回避行動がうまく学習できない

- より多くのエピソードを収集 (推奨: 20-30エピソード以上)
- **特に重要**: 衝突回避のエピソードを多めに記録
  - 衝突なしのエピソード: 衝突ありのエピソード = 1:1 程度が目安
- エポック数を増やす (`--epochs 200` 〜 `--epochs 500`)
- バッチサイズを調整 (`--batch-size 64`)

### BLE接続が不安定

- 制御周波数を下げる: `config.yaml` で `rate_hz: 50` または `30`
- 他のBluetooth機器との干渉を避ける
- toioとの距離を近づける(推奨: 2m以内)

## ライセンス

MIT

## 謝辞

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face のロボット学習フレームワーク
- [FaBo LeRobot Docs](https://faboplatform.github.io/LeRobotDocs/) - LeRobot環境構築ガイド・作業メモ
- [toio](https://toio.io/) - Sony の小型教育ロボット
