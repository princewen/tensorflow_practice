import os
import sys
import random

import tempfile
from subprocess import call


def main(file, temporary=False):
    tf_os, tpath = tempfile.mkstemp(dir='/home/mouna.mn/code/DIN-V2-CODE')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
    else:
        fd = open(file + '.shuf', 'w')

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    return fd


if __name__ == '__main__':
    main(sys.argv[1])

