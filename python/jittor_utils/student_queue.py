from socketserver import ThreadingTCPServer
import socket

def handle_connect(req:socket.socket, c_addr, server):
    print("get connect", c_addr, req)
    while True:
        buf = req.recv(2048)
        
        print(buf)

server = ThreadingTCPServer(("127.0.0.1", 8900), handle_connect)
server.serve_forever()