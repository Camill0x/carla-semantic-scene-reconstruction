import zmq


def create_latest_subscriber(context: zmq.Context, address: str) -> zmq.Socket:
    socket: zmq.Socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.connect(address)
    return socket


def create_latest_publisher(context: zmq.Context, address: str) -> zmq.Socket:
    socket: zmq.Socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(address)
    return socket


def drain_latest(socket: zmq.Socket):
    message = socket.recv_pyobj()
    while True:
        try:
            message = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return message
