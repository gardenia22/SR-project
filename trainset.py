__author__ = 'gardenia'
import os
import fnmatch

raw_path = "s1/"
align_path = "align/"
data_path = "data/"

def process_align_file(file):
    with open(file, "r") as f:
        text = []
        for line in f:
            word = line.strip().split(" ")[-1]
            if word != "sil":
                text.append(word)
    return " ".join(text)
with open(data_path + "align_text", "w") as out:
    for root, dirnames, filenames in os.walk(align_path):
        for filename in fnmatch.filter(filenames, "*.align"):
            align_file = os.path.join(root, filename)
            sentence = process_align_file(align_file)
            out.write("%s\n" % sentence)


