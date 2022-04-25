#!/usr/bin/env python3
#coding:utf-8

import sys
import os
import time
import subprocess
import argparse

# python3 run_predict.py --num=2 --datafile=1k --batch_size=1 --logfile=test/multitask_20211013.txt


parser = argparse.ArgumentParser(description='multitask test')
parser.add_argument('--num', type=int, default=2, help='number task')
parser.add_argument('--datafile', type=str, default="1k", help='datafile')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--logfile', type=str, default="", help='logfile')
args = parser.parse_args()
n = args.num
datafile = args.datafile
batch_size = args.batch_size
logfile = args.logfile

#cmd = './run_n.sh %s' % n
cmd = 'python3 predict.py --datafile=report/data_%s.txt --batch_size=%d --logfile=%s' % (datafile, batch_size, logfile)
#print('cmd:', cmd)

start = time.time()

#os.system(cmd)
pp=[]
for i in range(n):

    p = subprocess.Popen(cmd, shell=True)
    pp.append(p)

for p in pp:
    p.wait()    

predict_time = (time.time() - start)*1000
print('-'*40)
print('toal time:%.3f ms' % predict_time )

if __name__ == '__main__':
    pass

