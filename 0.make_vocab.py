import pandas as pd
train_df = pd.read_csv("data/train_set_small.csv", sep='\t')
print("4.字符统计")
from collections import Counter
all_lines = ' '.join(list(train_df["text"]))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
print(len(word_count))
print(word_count[:5])
with open("bert-mini/vocab.txt", "w", encoding="utf-8") as fout:
    fout.writelines("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    for word, count in word_count:
        fout.writelines(word + "\n")

print("make vocab.txt done : bert-mini/vocab.txt")