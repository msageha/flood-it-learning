# flood-it-learning

ban{n}/
ディレクトリには，縦横がn*nについての盤面が入っている

ban{n}/ban0~9　には，1000 * 10のランダムな盤面，
ban{n}/ans0~9 には，それぞれの盤面に対して，最短の手数と，それに至る経路が入っている．

make-ban.py　は，n*nのランダムな盤面を作成するスクリプト

solve.cpp　は，入力された盤面に対して，深さ優先探索を行い，最短の手数と，それに至る経路を出力するスクリプト

solve.py　は，盤面を入力し，数字を入れていくことで，盤面を変えていくスクリプト

run.sh は，並列処理にて，solve.cpp を走らせるスクリプト
