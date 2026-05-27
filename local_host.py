#!/usr/bin/env python3
"""
Safe no-cache local static server.

Usage:
    python serve_nocache.py
    python serve_nocache.py --port 8003
    python serve_nocache.py --host 127.0.0.1 --port 8003
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import os
import socket
import sys


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) != 0


def main():
    parser = argparse.ArgumentParser(description="Launch a no-cache local static server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to. Use 0.0.0.0 only if needed.")
    parser.add_argument("--port", type=int, default=8003, help="Port to serve on.")
    parser.add_argument("--dir", default=".", help="Directory to serve.")
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()

    if not serve_dir.exists() or not serve_dir.is_dir():
        raise SystemExit(f"Error: directory does not exist: {serve_dir}")

    if not port_available(args.host, args.port):
        raise SystemExit(
            f"Error: {args.host}:{args.port} is already in use.\n"
            f"Either stop the existing server or run with a different port."
        )

    os.chdir(serve_dir)

    server = ThreadingHTTPServer((args.host, args.port), NoCacheHandler)

    print(f"Serving directory: {serve_dir}")
    print(f"URL: http://{args.host}:{args.port}/")
    print("Caching disabled. Use Cmd+Shift+R if the browser still shows stale assets.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()