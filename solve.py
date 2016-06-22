# coding: utf-8
import sys
import random

num=int(raw_input())
ban = []
for i in range(num):
  ban.append(raw_input().split())

while(1):
  color=raw_input()
  if color == 0:
    break
  else:
    temp = ban[0][0]
    ban[0][0] = 0
    for k in range(num):
      for i in range(num):
        for j in range(num):
          if ban[i][j]==0:
            if i+1 != num and ban[i+1][j]==temp:
              ban[i+1][j]=0
            if i != 0 and ban[i-1][j] ==temp:
              ban[i-1][j]=0
            if j+1 != num and ban[i][j+1]==temp:
              ban[i][j+1]=0
            if j != 0 and ban[i][j-1]==temp:
              ban[i][j-1]=0

    for k in range(num):
      for i in range(num):
        for j in range(num):
          if ban[i][j]==0:
            if i+1 != num and ban[i+1][j]==color:
              ban[i+1][j]=0
            if i != 0 and ban[i-1][j] ==color:
              ban[i-1][j]=0
            if j+1 != num and ban[i][j+1]==color:
              ban[i][j+1]=0
            if j != 0 and ban[i][j-1]==color:
              ban[i][j-1]=0

    for i in range(num):
      for j in range(num):
        if ban[i][j]==0:
          ban[i][j] = color
      print ban[i]
