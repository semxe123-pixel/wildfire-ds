import os
import hashlib
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
AUTH_KEY = os.getenv("KMA_AUTH_KEY")

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
SAVE_DIR = "data/bronze/stn"
os.makedirs(SAVE_DIR, exist_ok=True)

TMS = ["202401010000", "202512312300"]     # 수집 기간 양 끝


def fetch(tm):
    params = {"inf": "AWS", "stn": "", "tm": tm, "help": 1, "authKey": AUTH_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if "#7777END" not in text:
        raise ValueError(f"불완전한 응답: {text[:80]!r}")
    return text


def main():
    for tm in TMS:
        text = fetch(tm)
        fname = f"stn_aws_{tm}.txt"
        with open(os.path.join(SAVE_DIR, fname), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"저장: {fname} ({len(text.splitlines())}줄)")


if __name__ == "__main__":
    main()