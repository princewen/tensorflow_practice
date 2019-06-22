from __future__ import print_function
import random

fi = open("local_test", "r")
ftrain = open("local_train_splitByUser", "w")
ftest = open("local_test_splitByUser", "w")

while True:
    rand_int = random.randint(1, 10)
    noclk_line = fi.readline().strip()
    clk_line = fi.readline().strip()
    if noclk_line == "" or clk_line == "":
        break
    if rand_int == 2:
        print(noclk_line, file=ftest)
        print(clk_line, file=ftest)
    else:
        print(noclk_line, file=ftrain)
        print(clk_line, file=ftrain)
        

