#!/usr/bin/env python3
#coding:utf-8


from sanic import Sanic, response
from sanic.websocket import WebSocketProtocol
import os
import sys
from clib import *

ssdb = newSSDB()

app = Sanic(__name__)

@app.websocket('/feed')
async def feed(request, ws):
    while True:
        dat = await ws.recv()
        print('Received: ' + dat)
        ssdb.qpush(queue_in, dat)

if __name__ == '__main__':
    # print('ssdb:', ssdb.info())
    app.run(host="0.0.0.0", port=8000, workers=1, debug=True) #, protocol=WebSocketProtocol



