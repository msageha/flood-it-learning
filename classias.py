# coding: utf-8
import sys



while(1):
  # try:
  #   n = int(raw_input())
  # except:
  #   break
  # ban = ""
  # for i in range(n):
  #   temp = raw_input().split()
  #   for j in range(n):
  #     ban = ban + " " + str(i) + "-" + str(j) + "-" + temp[j]

  try:
    n = int(raw_input())
  except:
    break
  ban = []
  for i in range(n):
    ban.append(raw_input().split())

  output = ""
  # for l in range(5):
  for i in range(5):
      for j in range(n):
        for k in range(n):
          output = output + " " + str(j) + "-" + str(k) + "-"
          if int(ban[j][k]) == i+1:
            output += str(i+1)
          else:
            output += '0'

    # for m in range(n-1):
    #   for n in range(n-1):
    #     ban[m][n] = str(int(ban[m][n]) - 1)
    #     if ban[m][n] == '0':
    #       ban[m][n] = "5"


  tesu = int(raw_input().split()[2])
  how = int(raw_input().split()[2])

  ans = [0,0,0,0,0,0]
  for i in range(how):
    ans[int(raw_input()[0])] += 1

  mx = max(ans)
  for i in range(6):
    if ans[i] == mx:
      mx = i
      break

  print str(mx) + " " + output
