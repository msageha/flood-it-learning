#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <stack>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <string>
#include <cstring>
#include <ctime>

using namespace std;

//debug
#define dump(x)  cerr << #x << " = " << (x) << endl;
#define print(x)  cout << #x << " = " << (x) << endl;

int tesu = 0;
vector <stack <int> > how_to;
int min_tesu = 99999; //最も最小の手数
stack <int> ans;

int initialize(vector<vector <int> > &ban, int now_color, int x, int y, int max, vector<bool> &change_color) { //現在の自分の持っているマスを-1に．
  if(x-1!=-1 && (ban[x-1][y]==now_color || ban[x-1][y]==0)) {
    ban[x-1][y]=-1;
    initialize(ban, now_color,x-1, y, max, change_color);
  }else{
    if (x-1!=-1 && ban[x-1][y]!=-1) change_color[ban[x-1][y]] = true;
  }
  if(x+1!=max && (ban[x+1][y]==now_color || ban[x+1][y]==0)) {
    ban[x+1][y]=-1;
    initialize(ban, now_color,x+1, y, max, change_color);
  }else{
    if (x+1!=max && ban[x+1][y] != -1) change_color[ban[x+1][y]] = true;
  }
  if(y-1!=-1 && (ban[x][y-1]==now_color || ban[x][y-1]==0)) {
    ban[x][y-1]=-1;
    initialize(ban, now_color,x, y-1, max, change_color);
  }else{
    if (y-1!=-1 && ban[x][y-1] != -1) change_color[ban[x][y-1]] = true;
  }
  if(y+1!=max && (ban[x][y+1]==now_color || ban[x][y+1]==0)) {
    ban[x][y+1]=-1;
    initialize(ban,now_color,x,y+1,max, change_color);
  }else{
    if (y+1!=max && ban[x][y+1] != -1) change_color[ban[x][y+1]] = true;
  }
  return 0;
}

bool eval(vector<vector <int> > &ban, int now_color, int max, vector<bool> &change_color) { //現在の自分の持っているますの更新
  ban[0][0] = -1;
  initialize(ban, now_color, 0, 0, max, change_color);
  bool check = true;
  for(int i=0;i<max;i++) {
    for(int j=0;j<max;j++) {
      if (ban[i][j]==-1) ban[i][j]=0;
      else check = false;
    }
  }

  return check;
}

void dfs(vector<vector <int> > ban, int now_color, int max) {
  vector<bool> change_color(6, false);
  bool check = eval(ban, now_color, max, change_color);
  if(check) {
    if (min_tesu>tesu) {
      min_tesu = tesu;
      how_to.clear();
      how_to.push_back(ans);
    }else if(min_tesu==tesu) {
      how_to.push_back(ans);
    }
  }else{
    if (min_tesu<=tesu) {
    }else {
      for (int i=1; i<6; i++) {
        if(change_color[i]) {
          change_color[i] = false;
          ans.push(i);
          tesu++;
          dfs(ban, i, max);
          ans.pop();
          tesu--;
        }
      }
    }
  }
}

int main() {
  int num; //盤面のサイズ
  int now_color; //今の自分の色
  while(cin>>num) {
    min_tesu = 99999;
    how_to.clear();
    while(!ans.empty()) ans.pop();
    vector<vector <int> > ban(10, vector<int>(10));
    cout << num << endl;
    for(int i=0;i<num;i++) { //盤面の入力
      for(int j=0;j<num;j++) {
        cin >> ban[i][j];
        cout << ban[i][j] << " ";
      }
      cout << endl;
    }
    now_color = ban[0][0];
    dfs(ban, now_color, num);
    cout << "tesu = " << min_tesu << endl;
    cout << "how = " << how_to.size() << endl;
    for(int i=0; i<how_to.size();i++) {
      string te = "";
      for(int j=0; j<min_tesu; j++) {
        te = to_string(how_to[i].top()) + " " + te;
        how_to[i].pop();
      }
      te.erase(te.size()-1);
      cout << te << endl;
    }
  }
  return 0;
}