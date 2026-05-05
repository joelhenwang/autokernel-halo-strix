#!/usr/bin/env python3
"""Quick TCP connectivity test for DDP over thunderbolt."""
import socket, sys, time

mode = sys.argv[1] if len(sys.argv) > 1 else "client"
host = sys.argv[2] if len(sys.argv) > 2 else "10.77.0.1"
port = int(sys.argv[3]) if len(sys.argv) > 3 else 29504

if mode == "server":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    print(f"LISTENING on 0.0.0.0:{port}", flush=True)
    s.settimeout(30)
    try:
        conn, addr = s.accept()
        print(f"CONNECTED from {addr}", flush=True)
        conn.sendall(b"OK")
        conn.close()
    except socket.timeout:
        print("TIMEOUT waiting for connection", flush=True)
    s.close()
elif mode == "client":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect((host, port))
        data = s.recv(16)
        print(f"CONNECTED to {host}:{port}, got: {data}")
        s.close()
    except Exception as e:
        print(f"FAILED: {e}")
        s.close()
