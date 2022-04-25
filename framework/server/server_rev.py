#!/usr/bin/env python3
#coding:utf-8


import os
import sys
from sanic import Sanic, response
from sanic.websocket import WebSocketProtocol
from clib import *

ssdb = newSSDB()

app = Sanic(__name__)

@app.websocket('/rev')
async def rev(request, ws):
    while True:
        size = ssdb.qsize(queue_out)
        if size > 0:
            data = ssdb.qpop(queue_out).decode()
            print('Sending: ' + data)
            await ws.send(data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, workers=1, debug=True)
