# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from socketserver import ThreadingTCPServer
import socket
import os
import sys
import threading

key_queue = {}

def handle_connect(req:socket.socket, c_addr, server):
    print("get connect", c_addr, req)
    skey = req.recv(1024).decode()
    print("get skey", skey)
    with lock:
        if skey not in key_queue:
            key_queue[skey] = []
        queue = key_queue[skey]
        queue.append(req)

        req.send(str(len(queue)-1).encode())
    while True:
        buf = req.recv(1024).decode()
        print(buf)
        with lock:
            if len(buf) == 0:
                for i,r in enumerate(queue):
                    if r is req:
                        for j in range(i+1, len(queue)):
                            queue[j].send(str(j-1).encode())
                        del queue[i]
                        print("queue size", len(queue))
                        break
                break
        

def wait_queue():
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 8900))
    s.sendall(skey.encode())
    while True:
        buf = s.recv(1024).decode()
        if len(buf) == 0:
            print("Cannot connect to queue server, please report this issue to admin.")
            sys.exit(1)
        if buf == '0':
            print("Begin")
            os.system(f"sleep {os.environ.get('SWAIT', '60')} && bash -c ' if kill -9 {os.getpid()} 2>/dev/null; then echo Timeout; fi; ' &")
            break
        else:
            print("Pending", buf)



if "SKEY" in os.environ:
    skey = os.environ["SKEY"]
    wait_queue()
else:
    lock = threading.Lock()
    server = ThreadingTCPServer(("127.0.0.1", 8900), handle_connect)
    server.serve_forever()
