# 観測量の離散化
- np.linspace で雑に等分割で離散化すると、逆に学習が上手く進まない。0付近の離散化を細かくしたらどうか？


# type1
500が高確率で出るようになる設定
学習回数は10000
再現性が低い。同じ設定でも複数回やると500rewardsにそもそも到達しないことがある

## 設定値

### Q学習の設定値（これらの設定値が妥当だとは限らない）
EPS = 0.2 # ε-greedyにおけるε 徐々に減らしたりするといいかも
LEARNING_RATE = 0.05 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率

### 観測量の離散化　ここも工夫の余地あり
    cart_pos = np.digitize(observation[0], bins=[-4.8, -3.2 ,-1.6, 0, 1.6, 3.2, 4.8]) # 1次元目
    cart_vel = np.digitize(observation[1], bins=[-6.0, -3.0, -1.5, -0.5, 0, 0.5, 1.5, 3.0, 6.0]) # 2次元目
    pole_ang = np.digitize(observation[2], bins=[-0.4, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.4]) # 3次元目
    pole_vel = np.digitize(observation[3], bins=[-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]) # 4次元目

if game_ID % 100 == 0:
            EPS *= 0.99 # εを徐々に減少させる

# type2
type1よりも下限が押しあがっている


## 設定値
type1のEPSの収束性を0.99から0.995にした