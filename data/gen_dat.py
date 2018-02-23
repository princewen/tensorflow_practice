#日丽风和人乐 国强民富年丰

source = open("source.txt",'w')
target = open("target.txt",'w')

with open("对联.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        print(line)
        source.write(line[0]+'\n')
        target.write(line[1]+'\n')