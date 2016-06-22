# coding: utf-8
import sys
import random

x = int(raw_input())
y = x
color = 5

for i in range(100):
  ban = []
  for j in range(x):
    temp = []
    for k in range(y):
      temp.append(str(random.randint(0, 99999)%5+1))
    ban.append(temp)

  print x
  for temp in ban:
    print " ".join(temp)

