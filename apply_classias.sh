cat ./ban5/ans* | python classias.py > temp.classias

head -n 9000 temp.classias | tail -n 1000 > dev.classias
tail -n 1000 temp.classias > test.classias
head -n 8000 temp.classias > train.classias
rm temp.classias

classias-train -m classias.model train.classias
classias-tag -qt -m classias.model < dev.classias
classias-tag -qt -m classias.model < test.classias
