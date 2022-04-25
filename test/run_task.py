#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import sys
import os
import time
import subprocess
import argparse


parser = argparse.ArgumentParser(description='multitask test')
parser.add_argument('--num', type=int, default=2, help='number task')
parser.add_argument('--datafile', type=str, default="8", help='datafile')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--logfile', type=str, default="", help='logfile')
args = parser.parse_args()
n = args.num
datafile = args.datafile
batch_size = args.batch_size
logfile = args.logfile

cmd = 'python3 ECTrans_task.py --cmd=client --datafile=report/data_%s.txt --batch_size=%d' % (datafile, batch_size)

# 开始计时
start = time.time()
pp=[]
for i in range(n):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    pp.append(p)

for p in pp:
    p.wait()    

predict_time = (time.time() - start)*1000
print('-'*40)
print('批量测试总计用时:%.3f 毫秒' % predict_time )

if __name__ == '__main__':
    pass

