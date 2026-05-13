from typing import Any

import zmq

ZmqContext = zmq.Context[zmq.Socket[bytes]]
ZmqSocket = zmq.Socket[bytes]


def create_latest_subscriber(context: ZmqContext, address: str) -> ZmqSocket:
    socket: ZmqSocket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.connect(address)
    return socket


def create_latest_publisher(context: ZmqContext, address: str) -> ZmqSocket:
    socket: ZmqSocket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(address)
    return socket


def drain_latest(socket: ZmqSocket) -> Any:
    message = socket.recv_pyobj()
    while True:
        try:
            message = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return message
