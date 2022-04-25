#!/usr/bin/env python3
#coding:utf-8


import os
import sys
from clib import *
from multiprocess import Process


def dowork():  #ssdb
    import time
    import random
    from clib import newSSDB

    SSDB_IP = '127.0.0.1'
    SSDB_PORT = 8888
    queue_in = 'queue_pipeline_in'
    queue_out = 'queue_pipeline_out'

    ssdb = newSSDB()
    if ssdb is None:
        print('SSDB can not connected...')
        return 0
    else:
        print('SSDB connected...')

    while True:
        size = ssdb.qsize(queue_in)
        if size > 0:
            data = ssdb.qpop(queue_in).decode()
            time.sleep(2)
            
            l = random.randrange(2,5)
            for i in range(l):
                dat = '%s,%s' % (data, time.time())
                ssdb.qpush(queue_out, dat)


if __name__ == '__main__':
    p = Process(target=dowork)#, args=(ssdb,)
    p.start()
