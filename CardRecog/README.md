## networks.py

使用するニューラルネットワークのネットワーク構造が記載されているファイルです．

## square_detectors.py

トランプカードが長方形であることを前提として，カメラ映像中からカード領域（長方形領域）を検出するコードが記載されています．  
このファイルは実行対象ではありません．

## CardRecog_train.py

数字13種類×スート4種類の52クラス分類を実行する単一の認識モデルを学習するプログラム．  
学習に使用するデータセットは 19, 20 行目で指定します．  
また，学習結果のモデルパラメータの保存先は 26, 27 行目で指定します．

**コマンド例**
```
python CardRecog_train.py --gpu 0 --epochs 20 --batchsize 256 --autosave
```
**オプション**
- gpu
  - 使用するGPUのID (-1を指定するとCPU上で動作します)
  - このオプションを指定しない場合，デフォルト値として -1 がセットされます．
  - cudaを使用できない環境では無視されます．
- epochs
  - 何エポック分学習するか
  - このオプションを指定しない場合，デフォルト値として 20 がセットされます．
- batchsize
  - バッチサイズ
  - このオプションを指定しない場合，デフォルト値として 256 がセットされます．
- autosave
  - 指定すると毎エポック終了時にモデルパラメータが自動保存されるようになります．
  - 保存先は ./CNN_models/autosaved_model_epX.pth です（ X はエポック番号 ）．

## CardRecog_test.py

上記の CardRecog_train.py で学習した認識モデルを用いて実際にカードの種類・スートを認識するプログラム．  
カメラを駆動してリアルタイムに認識処理を実行しますので，事前にUSBカメラをPCに接続しておく必要があります．

**コマンド例**
```
python CardRecog_test.py --gpu 0
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて CardRecog_train.py の同名オプションと同じです．

## CardRecog_train2.py

まず絵札か否かを判定し，絵札の場合は3種類(J,Q,K)×スート4種類の12クラス分類を，非絵札の場合は数字10種類×スート4種類の40クラス分類をそれぞれ実行する，
という形の2段階認識モデルを習するプログラム．  
学習に使用するデータセットは 19, 20 行目で指定します．  
また，学習結果のモデルパラメータの保存先は 26～29 行目で指定します．

**コマンド例**
```
python CardRecog_train2.py --gpu 0 --epochs 20 --batchsize 256 --autosave
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて CardRecog_train.py の同名オプションと同じです．
- epochs
  - 何エポック分学習するか
  - デフォルト値も含めて CardRecog_train.py の同名オプションと同じです．
- batchsize
  - バッチサイズ
  - デフォルト値も含めて CardRecog_train.py の同名オプションと同じです．
- autosave
  - 指定すると毎エポック終了時にモデルパラメータが自動保存されるようになります．
  - 保存先は ./CNN_models/{autosaved_cc_model_epX.pth, autosaved_pcc_model_epX.pth, autosaved_ncc_model_epX.pth} です（ X はエポック番号 ）．
    - autosaved_cc_model_epX.pth: 絵札か否かを判定するモデル
    - autosaved_pcc_model_epX.pth: 絵札に関する12クラス分類を行うモデル
    - autosaved_ncc_model_epX.pth: 非絵札に関する40クラス分類を行うモデル

## CardRecog_test2.py

## CNN_models

CardRecog_train.py および CardRecog_train2.py による学習結果の保存先として使用する想定のフォルダ．  
動作テスト時に作成したファイルが残っています．
