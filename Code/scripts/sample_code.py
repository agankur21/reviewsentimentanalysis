import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../common"))
sys.path.append(os.path.join(os.getcwd(), "../utils"))
complete_file_path = os.path.join(os.getcwd(), "../../Data", "reviews_sentiment_aspects_82000_temp_100000.txt")
count = -1
set_aspects = set([])
with open(complete_file_path, 'r') as f:
    for line in f:
        count += 1
        if count % 2 == 0:
            continue
        aspects = line.strip().split(",")
        set_aspects = set_aspects.union(set(aspects))
output_file_path = os.path.join(os.getcwd(), "../../Data", "reviews_aspects.txt")
out_file=open(output_file_path,'w')
for aspect in set_aspects:
    out_file.write(aspect+"\n")
out_file.close()
