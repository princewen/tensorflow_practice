from __future__ import print_function
import sys
import random
import time

def process_meta(file):
    fi = open(file, "r")
    fo = open("item-info", "w")
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)

def process_reviews(file):
    fi = open(file, "r")
    user_map = {}
    fo = open("reviews-info", "w")
    for line in fi:
        obj = eval(line)
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        rating = obj["overall"]
        time = obj["unixReviewTime"]
        print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), file=fo)

def manual_join():
    f_rev = open("reviews-info", "r")
    user_map = {}
    item_list = []
    for line in f_rev:
        line = line.strip()
        items = line.split("\t")
        #loctime = time.localtime(float(items[-1]))
        #items[-1] = time.strftime('%Y-%m-%d', loctime)
        if items[0] not in user_map:
            user_map[items[0]]= []
        user_map[items[0]].append(("\t".join(items), float(items[-1])))
        item_list.append(items[1])
    f_meta = open("item-info", "r")
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]
            arr = line.strip().split("\t")
    fo = open("jointed-new", "w")
    for key in user_map:
        sorted_user_bh = sorted(user_map[key], key=lambda x:x[1])
        for line, t in sorted_user_bh:
            items = line.split("\t")
            asin = items[1]
            j = 0
            while True:
                asin_neg_index = random.randint(0, len(item_list) - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == asin:
                    continue 
                items[1] = asin_neg
                print("0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg], file=fo)
                j += 1
                if j == 1:             #negative sampling frequency
                    break
            if asin in meta_map:
                print("1" + "\t" + line + "\t" + meta_map[asin], file=fo)
            else:
                print("1" + "\t" + line + "\t" + "default_cat", file=fo)


def split_test():
    fi = open("jointed-new", "r")
    fo = open("jointed-new-split-info", "w")
    user_count = {}
    for line in fi:
        line = line.strip()
        user = line.split("\t")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    fi.seek(0)
    i = 0
    last_user = "A26ZDKC53OP6JD"
    for line in fi:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 2:  # 1 + negative samples
                print("20180118" + "\t" + line, file=fo)
            else:
                print("20190119" + "\t" + line, file=fo)
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 2:
                print("20180118" + "\t" + line, file=fo)
            else:
                print("20190119" + "\t" + line, file=fo)
        i += 1

process_meta(sys.argv[1])
process_reviews(sys.argv[2])
manual_join()
split_test()
