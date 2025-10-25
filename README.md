# LeRobot × toio 行動クローニングフレームワーク

toio Core Cubeを行動クローニング（Behavior Cloning）で自律制御するための汎用フレームワークです。

## システムの特徴

**汎用的な行動学習基盤**
- Web UIを使った直感的なテレオペレーション
- スマホ対応のジョイスティックでデモンストレーションを記録
- 記録したデータから行動パターンを学習
- 様々なタスクに応用可能（追従、パターン走行、衝突回避など）

**シンプルな機械学習アプローチ**
- PyTorchによる行動クローニング（シンプルな3層MLP）
- 観測空間と行動空間を自由に設計可能
- CPU のみで訓練可能（数秒〜数十秒）

**2つのデータ収集方法**
- **手動収集**: Web UIでテレオペレーションしてデモンストレーションを記録
- **合成データ**: スクリプトで理想的なパターンを生成

**実装例：自律的な障害物回避**
- 前進→衝突検知→後退→約45度回転→前進を繰り返すRoomba風の動作
- 観測: `[collision, rotation_direction, frame_count]` (3次元)
- 行動: `[left_motor, right_motor]` (2次元、範囲 -100〜100)
- 衝突検知がピーキーなため、合成データで訓練した学習済みモデル（`models/policy.example.pth`）を同梱

## セットアップ

### 依存関係のインストール

[uv](https://github.com/astral-sh/uv) を使用します：

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトのセットアップ
uv sync
source .venv/bin/activate
```

### 設定ファイルの準備

```bash
cp config.example.yaml config.yaml
```

必要に応じて `config.yaml` を編集：

```yaml
robot:
  mac_address: null  # toioのMACアドレス (自動検出する場合はnull)
  collision_threshold: 3  # 衝突検知感度 (1-10: 小さいほど敏感)
```

## 使い方

### 学習済みモデルで自律制御

同梱の学習済みモデル (`models/policy.example.pth`) を使用：

```bash
uv run python scripts/inference.py models/policy.example.pth
```

toioは以下のパターンで動作します：
1. 前進（速度40）
2. 壁に衝突検知
3. 後退（10フレーム = 約0.17秒）
4. 左右どちらかに約45度回転（12フレーム = 約0.2秒）
5. 1に戻る

Ctrl+Cで停止できます。

### 自分でモデルを訓練する

#### 方法1: Web UIで手動データ収集（推奨）

**1. WebSocketサーバーの起動**

```bash
uv run python scripts/run_server.py
```

ブラウザで http://localhost:8765 を開く
**💡 スマホのブラウザで操作するのがおすすめです！** (片手でジョイスティック操作が可能)

**2. Operatorの起動**

```bash
uv run python scripts/run_operator.py
```

toioに自動接続され、Webコントローラーで操作できます。

**3. データ収集**

1. Webコントローラーで「記録開始」ボタンをクリック
2. toioを操作してデモンストレーション
   - 前進、回転などの基本動作
   - 壁にぶつかった時の回避行動を多様に記録
3. 「記録終了」ボタンをクリック（エピソードが自動保存）
4. 必要な数だけエピソードを繰り返し記録（20エピソード以上推奨）

保存先: `./datasets/toio_dataset/`

**注意**: 衝突検知がピーキーで安定した学習データを得るのが難しい場合があります。その場合は方法2の合成データをお試しください。

#### 方法2: 合成データの生成（代替手段）

手動データ収集が難しい場合、スクリプトで理想的なパターンを生成できます：

```bash
uv run python scripts/generate_dummy_data.py \
  --episodes 20 \
  --forward-frames 60 \
  --backward-frames 10 \
  --rotation-frames 12 \
  --forward-speed 40 \
  --backward-speed 40 \
  --rotation-speed 40
```

生成されるデータ：
- 20エピソード（左右回転が交互）
- 各エピソード: 60フレーム前進 + 10フレーム後退 + 12フレーム回転
- 観測: `[collision, rotation_direction, frame_count]`
  - `collision`: 0.0（通常）/ 1.0（衝突中）
  - `rotation_direction`: -1.0（左回転）/ 1.0（右回転）/ 0.0（通常時）
  - `frame_count`: 0.0〜1.0（衝突後の経過フレーム、正規化済み）
- 行動: `[left_motor, right_motor]` (-40〜40の範囲)

保存先: `./datasets/toio_dataset/`

#### モデルの訓練

```bash
uv run python scripts/train.py ./datasets/toio_dataset \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output ./models/policy.pth
```

訓練のポイント：
- 衝突フレームに10倍の重みをかけてデータ不均衡に対処
- アクション値をデータの実際の範囲で正規化（-1〜1に変換）
- Loss < 0.001 が目標（Lossが下がらない場合はデータ生成パラメータを調整）

**同梱モデル（policy.example.pth）の訓練結果**:
- データセット: 30エピソード（合成データ、各82フレーム、計2460フレーム）
- エポック数: 100
- バッチサイズ: 32
- 学習率: 0.001
- Lossの推移:
  - Epoch 1: Loss ≈ 0.15
  - Epoch 10: Loss ≈ 0.01
  - Epoch 30: Loss ≈ 0.001
  - Epoch 100: Loss ≈ 0.0003（最終）
- 訓練時間: 約10秒（CPUのみ）

**期待される結果**: 同様のパラメータで訓練すれば、30エポック程度で Loss < 0.001 に到達するはずです。Loss が下がらない場合は、データセットの確認をお勧めします。

#### 訓練したモデルで推論

どちらの方法でデータを収集した場合も、訓練後は同じように実行：

```bash
uv run python scripts/inference.py ./models/policy.pth
```

## 観測空間の設計

### なぜframe_countが必要か

このプロジェクトで使用している**単純なMLPモデルは決定論的**です。同じ観測からは常に同じ行動が出力されます。

**問題**: `[collision=1.0, rotation_direction=1.0]` だけでは不十分
- 後退すべきか？回転すべきか？モデルは判断できない
- 時系列的な状態遷移（後退→回転）を表現できない
- 同じ観測が後退時と回転時の両方で現れるため、学習が困難

**解決**: `frame_count` を追加
- 衝突後の経過フレーム数を0.0〜1.0に正規化して観測に含める
- モデルは `frame_count` の値から「後退フェーズ」か「回転フェーズ」かを学習
- 推論時は単純にカウンターをインクリメントするだけ（時間計算や状態判定は不要）

例：
- `[1.0, 1.0, 0.1]` → 後退（衝突直後、frame_countが小さい）
- `[1.0, 1.0, 0.7]` → 右回転（後退完了後、frame_countが大きい）

**補足**: VAEやGaussian Policyを使った確率的なモデルでは異なるアプローチも可能ですが、このプロジェクトではシンプルさを優先しています。

## プロジェクト構成

```
lerobot-toio-webctrl/
├── README.md
├── pyproject.toml
├── config.yaml                    # 設定ファイル（git管理外）
├── config.example.yaml            # 設定ファイルのテンプレート
├── models/
│   ├── policy.example.pth         # 学習済みモデル（すぐ試せる）
│   └── policy.pth                 # 自分で訓練したモデル（git管理外）
├── datasets/                      # データセット（git管理外）
│   └── toio_dataset/
├── scripts/
│   ├── generate_dummy_data.py     # 合成データ生成
│   ├── train.py                   # モデル訓練
│   ├── inference.py               # 自律制御実行
│   └── replay_dataset.py          # データセット確認用
└── lerobot_operator/
    ├── toio_driver.py             # toio BLE制御ドライバー
    └── ...
```

## 設定

### config.yaml

```yaml
robot:
  mac_address: null
  name_prefix: "toio Core Cube"
  scan_timeout_sec: 10.0
  scan_retry: 3
  collision_threshold: 3

control:
  max_speed: 100
  deadzone: 0.08
  expo: 0.3
  slew_rate: 300
  rate_hz: 60
  invert_x: false
  invert_y: false
  rotation_gain: 1.0
```

重要なパラメータ：
- `collision_threshold`: 衝突検知の感度（1-10）
  - 小さいほど敏感（誤検知が増える）
  - 大きいほど鈍感（実際の衝突を見逃す）
  - 推奨: 3（デフォルト）

## トラブルシューティング

### toioが見つからない

- toioの電源が入っているか確認
- Bluetoothがオンになっているか確認
- `config.yaml` で `mac_address` を明示的に指定

### 訓練時にLossが下がらない

- データ生成パラメータを確認
  - `forward_speed`, `backward_speed`, `rotation_speed` が同じ値か
  - エピソード数が十分か（20以上推奨）
- エポック数を増やす（`--epochs 200`）

### 衝突検知が敏感すぎる/鈍すぎる

`config.yaml` の `collision_threshold` を調整：

```yaml
robot:
  collision_threshold: 2  # より敏感に（1〜10）
```

レベル2: より敏感（誤検知増）
レベル3: デフォルト（推奨）
レベル4: やや鈍感（見逃し増）

### 推論時に回転角度が大きすぎる/小さすぎる

データ生成時の `--rotation-frames` を調整して再訓練：

```bash
# より小さい角度（例: 30度）
uv run python scripts/generate_dummy_data.py --rotation-frames 8

# より大きい角度（例: 90度）
uv run python scripts/generate_dummy_data.py --rotation-frames 20
```

`rotation_frames=12` で約45度回転（推奨）

## 技術的な詳細

### 衝突回避における合成データの利点

このシステムには手動操作でデータを収集する機能（Web UI + テレオペレーション）も実装されています。しかし、**衝突回避のような高速な反応が必要な動作**では、合成データの方が優れている場合があります。

**衝突回避で手動データ収集が難しい理由**:
- toioの衝突検知がピーキー（感度調整が難しい）
- 人間の反応速度では衝突→後退→回転の一連の動作を安定して記録しにくい
- ジョイスティックの微小なブレがノイズとなる
- データの品質にばらつきが出やすい

**合成データのアプローチ**:
- 理想的な衝突回避パターンをスクリプトで生成
- 決定論的で再現性のあるデータセット
- Loss < 0.001 を安定して達成

**他のユースケースでは**: 追従行動、特定パターンの走行など、人間が操作しやすいタスクでは手動データ収集も有効です。

### モデルアーキテクチャ

シンプルな3層全結合NN：
- 入力: 3次元 `[collision, rotation_direction, frame_count]`
- 隠れ層: 128次元 (ReLU) → 128次元 (ReLU)
- 出力: 2次元 `[left_motor, right_motor]` (tanh、-1〜1に正規化)

最適化:
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- 衝突フレームに10倍の重み

### 推論時の動作

```python
# 推論ループ（簡略化）
collision_active = False
frame_count = 0

while True:
    if new_collision and not collision_active:
        collision_active = True
        frame_count = 0
        rotation_direction = random.choice([-1.0, 1.0])

    if collision_active:
        frame_count += 1
        if frame_count >= 22:  # 10 + 12
            collision_active = False

    obs = [
        1.0 if collision_active else 0.0,
        rotation_direction if collision_active else 0.0,
        frame_count / 21.0 if collision_active else 0.0
    ]

    action = model(obs)
    toio.move(action[0], action[1])
```

**重要**: 推論コードは状態遷移のロジックを持たない
- カウンターをインクリメントするだけ
- 「後退→回転」の切り替えはモデルが学習

## ライセンス

MIT

## 謝辞

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face のロボット学習フレームワーク
- [FaBo LeRobot Docs](https://faboplatform.github.io/LeRobotDocs/) - LeRobotでtoioを制御・学習する際の参考にさせていただきました。
- [toio](https://toio.io/) - Sony の小型教育ロボット
