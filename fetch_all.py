import os
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()
AUTH_KEY = os.getenv("KMA_AUTH_KEY")
SAVE_DIR = os.getenv("DATA_DIR", "data/bronze/aws")   # .env에 없으면 기본값
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"

# 수집 범위 (교수님 지정)
START = datetime(2024, 1, 1, 0)
END = datetime(2025, 12, 31, 23)


def hourly_range(start, end):
    """start부터 end까지 1시간 간격으로 'yyyymmddhhmi' 문자열을 하나씩 내놓는다."""
    t = start
    while t <= end:
        yield t.strftime("%Y%m%d%H%M")
        t += timedelta(hours=1)


def fetch_one(tm):
    """시각 하나를 받아 저장. 응답이 불완전하면 에러."""
    params = {"tm": tm, "help": 0, "authKey": AUTH_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if "#7777END" not in text:                       # 상태 200인데 본문이 에러인 경우 방어
        raise ValueError(f"불완전한 응답: {text[:80]!r}")
    path = os.path.join(SAVE_DIR, f"{tm}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fetch_all():
    tms = list(hourly_range(START, END))
    total = len(tms)
    print(f"대상 시각 {total}개, 저장 위치 {SAVE_DIR}")

    saved = skipped = failed = 0
    for i, tm in enumerate(tms, 1):
        path = os.path.join(SAVE_DIR, f"{tm}.txt")
        if os.path.exists(path):                     # 이미 받은 시각은 건너뜀
            skipped += 1
            continue

        for attempt in range(1, 4):                  # 최대 3번 시도
            try:
                fetch_one(tm)
                saved += 1
                break
            except Exception as e:
                print(f"[{tm}] 실패 {attempt}/3: {e}")
                time.sleep(5)
        else:                                        # 3번 다 실패하면 기록만 하고 다음으로
            failed += 1
            with open("failed_tm.txt", "a", encoding="utf-8") as f:
                f.write(tm + "\n")

        time.sleep(0.3)                              # 서버 부담 완화
        if i % 100 == 0:
            print(f"{i}/{total}  저장 {saved} / 건너뜀 {skipped} / 실패 {failed}")

    print(f"끝. 저장 {saved} / 건너뜀 {skipped} / 실패 {failed}")


if __name__ == "__main__":
    fetch_all()