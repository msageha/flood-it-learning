#include <stack>
#include <iostream>
#include <vector>

int main()
{
  using namespace std;

  stack<int> st;    // int型のスタック

  // 要素のプッシュとポップ
  st.push( 10 );
  st.push( 20 );
  // cout << st[0] << st[1] << endl;
  cout << st.top() << "をポップします" << endl;
  st.pop();
  return 0;
}