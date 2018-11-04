import sys
import hashlib
import random

fin = open("jointed-new-split-info", "r")
ftrain = open("local_train", "w")
ftest = open("local_test", "w")

last_user = "0"
common_fea = ""
line_idx = 0
for line in fin:
    items = line.strip().split("\t")
    ds = items[0]
    clk = int(items[1])
    user = items[2]
    movie_id = items[3]
    dt = items[5]
    cat1 = items[6]

    if ds=="20180118":
        fo = ftrain
    else:
        fo = ftest
    if user != last_user:
        movie_id_list = []
        cate1_list = []
        #print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + "" + "\t" + "" 
    else:
        history_clk_num = len(movie_id_list)
        cat_str = ""
        mid_str = ""
        for c1 in cate1_list:
            cat_str += c1 + ""
        for mid in movie_id_list:
            mid_str += mid + ""
        if len(cat_str) > 0: cat_str = cat_str[:-1]
        if len(mid_str) > 0: mid_str = mid_str[:-1]
        if history_clk_num >= 1:    # 8 is the average length of user behavior
            print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + mid_str + "\t" + cat_str
    last_user = user
    if clk:
        movie_id_list.append(movie_id)
        cate1_list.append(cat1)                
    line_idx += 1
