"""
Logger Utility
"""

from datetime import datetime


def info(message):
    print(f"[INFO {datetime.now().strftime('%H:%M:%S')}] {message}")


def warning(message):
    print(f"[WARNING {datetime.now().strftime('%H:%M:%S')}] {message}")


def error(message):
    print(f"[ERROR {datetime.now().strftime('%H:%M:%S')}] {message}")