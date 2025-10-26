# toio 行動クローニングフレームワーク

toio Core Cubeを行動クローニング（Behavior Cloning）で自律制御するための**学習・実験用フレームワーク**です。

## このプロジェクトについて

このプロジェクトは、ロボット制御における行動クローニング（Imitation Learning）のパイプライン全体を実機で学ぶための教育用フレームワークです。

### 現状と限界

同梱の衝突回避タスクは**非常にシンプル**で、実際には以下のようなルールベースのコードで十分実装できます：

```python
# 等価なルールベース実装（約10行）
if collision_detected:
    if frame < 10:
        return [-40, -40]  # 後退
    else:
        return [40, -40] if random() < 0.5 else [-40, 40]  # ランダム回転
else:
    return [40, 40]  # 前進
```

合成データ生成スクリプトも、本質的には上記のルールをデータ形式で記述し直しているだけです。つまり、**このタスクでは機械学習の真の強みを活かせていません**。

### フレームワークとしての価値

しかし、このプロジェクトには以下の学習価値があります：

- **完全なパイプライン**: データ収集→訓練→推論の一連の流れを実機で体験できる
- **実装の参考**: Web UI、BLE通信、データセット管理など、実用的なコンポーネント
- **拡張性**: より複雑なタスクへの拡張基盤として利用可能

### より意味のある応用例

このフレームワークは、以下のような複雑なタスクで真価を発揮します：

- **視覚ベースの制御**: カメラ画像からライントレースや物体追従
- **位置センサー活用**: toioの位置検出機能を使った経路学習
- **マルチエージェント**: 複数のキューブが協調して動作するパターン学習
- **連続的な制御**: 単純なif文では記述困難な滑らかな動作パターン

現在の衝突回避タスクは、これらのタスクを実装する前の**動作確認用デモ**として位置づけてください。

## フレームワークの特徴

**完全な行動クローニングパイプライン**
- Web UIを使った直感的なテレオペレーション
- スマホ対応のジョイスティックでデモンストレーションを記録
- 記録したデータから行動パターンを学習
- 訓練したモデルで実機を自律制御

**シンプルで拡張可能な設計**
- PyTorchによる行動クローニング（シンプルな3層MLP）
- 観測空間と行動空間を自由に設計可能
- CPU のみで訓練可能（数秒〜数十秒）
- より複雑なタスクへの拡張が容易

**2つのデータ収集方法**
- **手動収集**: Web UIでテレオペレーションしてデモンストレーションを記録
- **合成データ**: スクリプトでパターンを生成（シンプルなタスク用）

**デモタスク：衝突回避**

動作確認用のシンプルなデモとして、障害物回避タスクを実装：
- 前進→衝突検知→後退→約45度回転→前進を繰り返す
- 観測: `[collision, rotation_direction, frame_count]` (3次元)
  - `collision`: 0.0 (通常) / 1.0 (衝突中)
  - `rotation_direction`: -1.0 (左) / 1.0 (右) / 0.0 (通常)
  - `frame_count`: 0.0-1.0 (衝突後の経過、正規化済み)
- 行動: `[left_motor, right_motor]` (2次元、範囲 -100〜100)
- 合成データで訓練した学習済みモデル（`models/policy.example.pth`）を同梱

⚠️ **注意**: このタスクは単純すぎて機械学習の強みを活かせていません。より複雑なタスク（視覚制御、位置ベースの経路学習など）での使用を推奨します。

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

動作例は以下の動画の通りです。

https://github.com/user-attachments/assets/00f324d9-398a-4f42-9d9b-438f364d7229

toioは以下のパターンで動作します：
1. 前進（速度40）
2. 壁に衝突検知
3. 後退（10フレーム = 約0.17秒）
4. 左右どちらかに約45度回転（12フレーム = 約0.2秒）
5. 1に戻る

Ctrl+Cで停止できます。

### 自分でモデルを訓練する

#### 方法1: Web UIで手動データ収集（推奨）

**Operatorの起動**

```bash
uv run python scripts/operator.py
```

Webサーバーが起動し、toioに自動接続されます。
ブラウザで http://localhost:8765 を開いてコントローラーを表示
**💡 スマホのブラウザで操作するのがおすすめです！** (片手でジョイスティック操作が可能)

**データ収集**

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
- 観測: `[collision, rotation_direction, frame_count]` (3次元)
  - `collision`: 0.0（通常）/ 1.0（衝突中）
  - `rotation_direction`: -1.0（左回転）/ 1.0（右回転）/ 0.0（通常時）
  - `frame_count`: 0.0-1.0（衝突後の経過フレーム、正規化済み）
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

## プロジェクト構成

```
toio-bc/
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
│   ├── operator.py                # テレオペレーション（Webサーバー統合）
│   ├── train.py                   # モデル訓練
│   ├── inference.py               # 自律制御実行
│   ├── replay.py                  # データセット確認用
│   └── generate_dummy_data.py     # 合成データ生成
└── src/toio_bc/
    ├── __init__.py
    ├── operator.py                # テレオペレーション（メイン実装）
    ├── train.py                   # 訓練（メイン実装）
    ├── inference.py               # 推論（メイン実装）
    ├── replay.py                  # 再生（メイン実装）
    ├── core/                      # コアモジュール
    │   ├── toio_driver.py         # toio BLE制御ドライバー
    │   ├── episode_recorder.py    # データ記録
    │   └── mixing.py              # モーター制御ミキシング
    └── server/                    # Webサーバー静的ファイル
        └── static/
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

### データセット形式

このフレームワークは、シンプルなnpz形式でデータを保存します：

**ファイル構成**:
```
datasets/toio_dataset/
├── data.npz              # 観測と行動のnumpy配列
└── meta/
    ├── info.json         # データセットメタデータ
    └── episodes.json     # エピソード情報
```

**data.npz の内容**:
- `observation.state`: 観測ベクトル (float32, shape: [N, obs_dim])
- `action`: 行動ベクトル (float32, shape: [N, 2])
- `episode_index`: エピソード番号 (int64, shape: [N])
- `frame_index`: フレーム番号 (int64, shape: [N])
- `timestamp`: タイムスタンプ (float32, shape: [N])
- `next.done`: エピソード終了フラグ (bool, shape: [N])

この形式は、ロボット学習フレームワークで一般的に使用されるデータ構造に基づいています。

### 合成データの特徴と限界

このシステムには手動操作でデータを収集する機能（Web UI + テレオペレーション）も実装されていますが、デモタスクの衝突回避では合成データを使用しています。

**合成データの特徴**:
- 理想的な動作パターンをスクリプトで生成
- 決定論的で再現性のあるデータセット
- 訓練の安定性（Loss < 0.001 を確実に達成）

**重要な限界**:
- **本質的にはルールベースと同等**: 合成データ生成スクリプトは、ルールをデータ形式で記述し直しているだけ
- **機械学習の意義が薄い**: 単純なif文で書けるロジックをニューラルネットで近似しているに過ぎない
- **教育目的の位置づけ**: パイプラインの動作確認には有用だが、実用性は低い

**より意味のあるユースケース**:
- **視覚ベースの制御**: カメラ画像からの学習（ルールで記述困難）
- **人間のデモンストレーション**: 熟練者の微妙な制御を模倣
- **連続的な制御**: 滑らかな軌道生成など、if文では表現しにくい動作

このフレームワークの真価は、上記のような複雑なタスクで発揮されます。

### モデルアーキテクチャ

シンプルな3層全結合NN：
- 入力: 3次元 `[collision, rotation_direction, frame_count]`
- 隠れ層: 128次元 (ReLU) → 128次元 (ReLU)
- 出力: 2次元 `[left_motor, right_motor]` (tanh、-1〜1に正規化)

最適化:
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- 衝突フレームに10倍の重み

### frame_countの役割

このプロジェクトで使用している**決定論的なMLPモデル**では、同じ観測からは常に同じ行動が出力されます。

**問題**: `[collision=1.0, rotation_direction=1.0]` だけでは不十分
- 後退すべきか？回転すべきか？モデルは判断できない
- 時系列的な状態遷移（後退→回転）を表現できない

**解決**: `frame_count` を追加
- 衝突後の経過フレーム数を0.0〜1.0に正規化して観測に含める
- モデルは `frame_count` の値から「後退フェーズ」か「回転フェーズ」かを学習
- 推論時は単純にカウンターをインクリメントするだけ

例：
- `[1.0, 1.0, 0.1]` → 後退（衝突直後、frame_countが小さい）
- `[1.0, 1.0, 0.7]` → 右回転（後退完了後、frame_countが大きい）

**本質的には**: これはルールベースの状態管理をframe_countという形で表現しているだけです。

### なぜ機械学習を使うのか？

**率直な回答**: 現在の衝突回避タスクでは、**機械学習は不要**です。

以下のような単純なルールベースのコードで同等の動作が実現できます：

```python
# ルールベース版（機械学習不要）
if collision:
    if frame < 10:
        return [-40, -40]  # 後退
    elif frame < 22:
        return [40, -40] if direction > 0 else [-40, 40]  # 回転
else:
    return [40, 40]  # 前進
```

**それでも機械学習版を実装している理由**:

1. **学習目的**: 行動クローニングのパイプライン全体を体験できる
2. **実装の参考**: データ収集、訓練、推論の具体例を提供
3. **拡張の基盤**: より複雑なタスクへの拡張時に再利用可能

**機械学習が本当に必要になる例**:

- **高次元の入力**: カメラ画像（数万次元）からの制御
- **複雑なパターン**: 人間の暗黙知を含む微妙な制御
- **適応的な動作**: 環境の変化に応じた動作の調整

このプロジェクトは、そのような複雑なタスクを実装する前の**学習用フレームワーク**として利用してください。

### 推論時の動作

参考までに、現在の推論ループの構造：

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
        if frame_count >= 22:
            collision_active = False

    obs = [
        1.0 if collision_active else 0.0,
        rotation_direction,
        frame_count / 21.0  # 正規化されたカウンター
    ]
    action = model(obs)  # ← ここでニューラルネットを使用
    toio.move(action[0], action[1])
```

モデルは「衝突フラグ」「回転方向」「経過フレーム」から、適切なモーター出力を予測します。しかし、frame_countは本質的にはルールベースの状態管理と同じで、これは上記のif文と本質的に同じことをしています。

## ライセンス

MIT

## 参考資料

- [toio](https://toio.io/) - Sony の小型教育ロボット
