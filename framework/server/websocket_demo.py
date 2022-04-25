#!/usr/bin/env python3
#coding:utf-8


# from sanic.response import json
# from sanic.websocket import WebSocketProtocol
from sanic import Sanic, response
from sanic.response import file
from multiprocess import Process, Queue
import os
import sys
import asyncio

msg = Queue()
inmsg = Queue()
outmsg = Queue()

def dowork(inmsg, outmsg): #
    import time
    import random
    while True:
        dat = inmsg.get()
        time.sleep(2)
        #await asyncio.sleep(2)
        l = random.randrange(2,5)
        for i in range(l):
            outmsg.put('%s,%s' % (dat, time.time()) )
        print('sucess:%s' % dat)

app = Sanic(__name__)

@app.route('/')
async def index(request):
    return await file('websocket.html')

@app.route('/send', methods=['POST','GET'])
async def send(request):
    txt = request.get('text')
    if txt:
        print('txt:', txt)
        inmsg.pug(txt)
    res = {'result':'OK'}
    return response.json(res)

@app.websocket('/feed')
async def feed(request, ws):
    dat = await ws.recv()
    if dat:
        print('Received: ' + dat)
        inmsg.put(dat)

    while True:
        #while not outmsg.empty():
        data = outmsg.get()
        print('Sending: ' + data)
        await ws.send(data)

@app.websocket('/rev')
async def rev(request, ws):
    while True:
        data = outmsg.get()
        print('Sending: ' + data)
        await ws.send(data)

if __name__ == '__main__':
    #app.add_task(dowork()) #inmsg, outmsg

    p = Process(target=dowork, args=(inmsg, outmsg))
    p.start()
    #p.join()
    app.run(host="0.0.0.0", port=8000, workers=1, debug=True) # auto_reload=True
    #app1.run(host="0.0.0.0", port=8001, workers=1, debug=True)

    # app.run(host="0.0.0.0", port=8000, protocol=WebSocketProtocol)
    print('running...')
