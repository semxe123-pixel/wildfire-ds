import time
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

# 매달 1일 00시 24개 + 수집 기간의 마지막 시각 1개 = 25개
TMS = [f"{y}{m:02d}010000" for y in (2024, 2025) for m in range(1, 13)] + ["202512312300"]


def fetch(tm):
    params = {"inf": "AWS", "stn": "", "tm": tm, "help": 1, "authKey": AUTH_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if "#7777END" not in text:
        raise ValueError(f"불완전한 응답: {text[:80]!r}")
    return text


def main():
    for tm in TMS:
        fname = f"stn_aws_{tm}.txt"
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):                      # 이미 받은 시점은 건너뜀
            print(f"건너뜀(이미 있음): {fname}")
            continue

        for attempt in range(1, 4):                   # 최대 3번 시도
            try:
                text = fetch(tm)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"저장: {fname} ({len(text.splitlines())}줄)")
                break
            except Exception as e:
                print(f"[{tm}] 실패 {attempt}/3: {type(e).__name__}")
                time.sleep(5)
        else:                                         # 3번 다 실패하면 알리고 다음 시점으로
            print(f"[{tm}] 3번 모두 실패. 나중에 다시 실행할 것")

        time.sleep(0.3)                               # 서버 부담 완화        

if __name__ == "__main__":
    main()