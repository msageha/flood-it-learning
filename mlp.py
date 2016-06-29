#coding: utf-8
import numpy as np
# import matplotlib.pyplot as plt
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys
import ipdb

# plt.style.use('ggplot')

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100
# 学習の繰り返し回数
n_epoch   = 20
# 中間層の数
n_units   = 50

ban_arr = []
ans_arr = []
s = set()
for i in range(10):
  filename = "ban5/ans" + str(i) 
  with open(filename) as f:
    line = f.readline()
    while line:
      ban = []
      temp = []
      n = int(line)
      for j in range(n):
        temp.append(f.readline().split())
        for k in range(n):
          temp[j][k] = int(temp[j][k])

      for j in range(5):
        temp_ban = [[0 for a in range(n)] for b in range(n)]
        for a in range(n):
          for b in range(n):
            if temp[a][b] == j+1:
              temp_ban[a][b] = 1
        ban.append(temp_ban)

      ban_arr.append(ban)
      # ipdb.set_trace()

      temp2 = f.readline()
      tesu = int(temp2.split()[2])
      temp3 = f.readline()
      how = int(temp3.split()[2])

      ans = [0,0,0,0,0,0]
      for j in range(how):
        ans[int(f.readline()[0])] += 1

      mx = max(ans)
      for j in range(6):
        if ans[j] == mx:
          mx = j
          break
      mx = tesu #教師データを手数とするとき
      s.add(mx)
      ans_arr.append(mx)
      line = f.readline()

x_train = np.array(ban_arr, dtype=np.float32)[:8000]
y_train = np.array(ans_arr, dtype=np.int32)[:8000] - 1
x_test = np.array(ban_arr, dtype=np.float32)[8000:9000]
y_test = np.array(ans_arr, dtype=np.int32)[8000:9000] - 1
x_dev = np.array(ban_arr, dtype=np.float32)[9000:10000]
y_dev = np.array(ans_arr, dtype=np.int32)[9000:10000] - 1

N = len(x_train) #N このデータ
N_test = y_test.size #テストサイズ

#モデルの定義

# Prepare multi-layer perceptron model
# 多層パーセプトロンモデルの設定
# 入力 784次元、出力 10次元
model = FunctionSet(l1=F.Linear(5*n*n, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 10))

# Neural net architecture
# ニューラルネットの構造
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    h3 = F.dropout(F.relu(model.l2(h2)), train=train)
    y  = model.l3(h3)
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

l1_W = []
l2_W = []
l3_W = []

# Learning Loop
for epoch in xrange(1, n_epoch+1):
  print 'epoch', epoch

  # training

  sum_accuracy = 0
  sum_loss = 0

  # 0〜Nまでのデータをバッチサイズごとに使って学習
  for i in xrange(0, N, batchsize):
    x_batch = x_train[i:i+batchsize]
    y_batch = y_train[i:i+batchsize]

    # 勾配を初期化
    optimizer.zero_grads()

    # 順伝播させて誤差と精度を算出
    loss, acc = forward(x_batch, y_batch, train=False)

    # 誤差逆伝播で勾配を計算
    loss.backward()
    optimizer.update()

    train_loss.append(loss.data)
    train_acc.append(acc.data)
    sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
    sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

  # 訓練データの誤差と、正解精度を表示
  print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

  # evaluation
  # テストデータで誤差と、正解精度を算出し汎化性能を確認
  sum_accuracy = 0
  sum_loss     = 0
  for i in xrange(0, N_test, batchsize):
    x_batch = x_test[i:i+batchsize]
    y_batch = y_test[i:i+batchsize]

    # 順伝播させて誤差と精度を算出
    loss, acc = forward(x_batch, y_batch, train=False)

    test_loss.append(loss.data)
    test_acc.append(acc.data)
    sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
    sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # テストデータでの誤差と、正解精度を表示
  print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)


  # 学習したパラメーターを保存
  l1_W.append(model.l1.W)
  l2_W.append(model.l2.W)
  l3_W.append(model.l3.W)

# 精度と誤差をグラフ描画
# plt.figure(figsize=(8,6))
# plt.plot(range(len(train_acc)), train_acc)
# plt.plot(range(len(test_acc)), test_acc)
# plt.legend(["train_acc","test_acc"],loc=4)
# plt.title("Accuracy of digit recognition.")
# plt.plot()