# coding: utf-8
import sys
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import ipdb

# scikit-learn の Linear Regression を利用
from sklearn import linear_model

ban_arr = []
ans_arr = []
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
      mx = tesu #教師データを手数とするとき．
      ans_arr.append(mx)
      line = f.readline()

  # ipdb.set_trace()

x_train = np.array(ban_arr, dtype=np.float32)[:8000]
y_train = np.array(ans_arr, dtype=np.int32)[:8000] - 1
x_test = np.array(ban_arr, dtype=np.float32)[8000:9000]
y_test = np.array(ans_arr, dtype=np.int32)[8000:9000] - 1
x_dev = np.array(ban_arr, dtype=np.float32)[9000:10000]
y_dev = np.array(ans_arr, dtype=np.int32)[9000:10000] - 1

#reshape でデータ整形
x_train = x_train.reshape([x_train.shape[0], -1]) # -1 は，長さがx_train.shape[0]になるように勝手に埋めてくれる
y_train = y_train.reshape([y_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
y_test = y_test.reshape([y_test.shape[0], -1])
x_dev = x_dev.reshape([x_dev.shape[0], -1])
y_dev = y_dev.reshape([y_dev.shape[0], -1])

N = len(x_train) #N このデータ
N_test = y_test.size #テストサイズ


# x_combined_std = np.vstack((x_train_std, x_test_std)) # 縦に連結
# y_combined = np.hstack((y_train, y_test)) # 横に連結

### ロジスティック回帰
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#モデルの生成
lr = LogisticRegression(C=9000.0)
#学習
lr.fit(x_train, y_train)
#学習したモデルの精度
print 'モデルの精度:' + str(lr.score(x_train, y_train))

#モデルに伴う生存率の予測値
predict_y = lr.predict(x_test)

#各分類のどのクラスに分類されるかの確率
lr.predict_proba(x_test)

#実際の値と予測値の比率
print 'accuracy:' + str(accuracy_score(y_test, predict_y))
# 上とやってること同じ
# print sum(predict_y == y_test) * 1.0 / y_test.shape[0]


# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111, projection='3d')
# # ax = Axes3D(fig)

# # Data
# ax.scatter(x_train[:, 0], x_train[:, 1], y_train, facecolor='#00CC00')
# ax.scatter(x_test[:, 0], x_test[:, 1], y_test, facecolor='#FF7800')

# coef = lr.coef_
# line = lambda x1, x2: coef[0] * x1 + coef[1] * x2

# grid_x1, grid_x2 = np.mgrid[-2:2:10j, -2:2:10j]
# ax.plot_surface(grid_x1, grid_x2, line(grid_x1, grid_x2),
#                 alpha=0.1, color='k')
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.zaxis.set_visible(False)

# fig.savefig('image.png', bbox='tight')



# np.random.seed(1)

# def p_y_given_x(x, w, b):
#   # x, w, b から y の予測値 (yhat) を計算
#   def sigmoid(a):
#     return 1.0 / (1.0 + np.exp(-a))
#   return sigmoid(np.dot(x, w) + b)

# def grad(x, y, w, b):
#   # 現予測値から勾配を計算
#   error = y - p_y_given_x(x, w, b)
#   w_grad = -np.mean(x.T * error, axis=1)
#   b_grad = -np.mean(error)
#   return w_grad, b_grad

# def gd(x, y, w, b, eta=0.1, num=100):
#   for i in range(1, num):
#     # 入力をまとめて処理
#     w_grad, b_grad = grad(x, y, w, b)
#     w -= eta * w_grad
#     b -= eta * b_grad
#     e = np.mean(np.abs(y - p_y_given_x(x, w, b)))
#     yield i, w, b, e

# # w, b の初期値を作成
# w, b = np.zeros(2), 0
# gen = gd(x, y, w, b)

# # gen はジェネレータ
# gen
# # <generator object gd at 0x11108e5f0>

# # 以降、gen.next() を呼ぶたびに 一回 勾配降下法を実行して更新した結果を返す。
# # タプルの中身は (イテレーション回数, w, b, 誤差)
# gen.next()
# # (1, array([-0.027  , -0.06995]), 0.0, 0.47227246182037463)

# gen.next()
# # (2, array([-0.04810306, -0.12007078]), 0.0054926687253766763, 0.45337584157628485)
