import sys, re, string, argparse
from collections import OrderedDict

file = open("icp2.txt", "r")
words = file.read().split()
dict = {}

for word in words:
    pattern = re.compile('[\W]', re.IGNORECASE | re.UNICODE)
    word = pattern.sub('', word).lower();
    if word and word in dict:
        dict[word] += 1
    else:
        dict[word] = 1

sortedWords = OrderedDict(sorted(dict.items(), key=lambda x: x[1], reverse=True))

string = ''
for i in sortedWords:
    num = str(sortedWords[i])
    string = string + i + ', ' + num + "\n"

f = open("result.txt", "w")
f.write(string)