import os
import math
import time
import hashlib
from datetime import datetime
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
SERVICE_KEY = os.getenv("DATA_GO_KR_KEY")          # Decoding 키

BASE_URL = "https://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"
SAVE_DIR = "data/bronze/fire"
INDEX_PATH = "data/bronze/fire_index.parquet"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_ROWS = 99
MONTHS = [f"{y}{m:02d}" for y in (2024, 2025) for m in range(1, 13)]   # '202401' ... '202512'


def month_range(ym):
    """'202503' → ('20250301', '20250331')"""
    y, m = int(ym[:4]), int(ym[4:])
    last = (datetime(y + (m == 12), (m % 12) + 1, 1) - pd.Timedelta(days=1)).day
    return f"{ym}01", f"{ym}{last:02d}"


def fetch_page(st, ed, page):
    """한 페이지 요청. (응답 텍스트, resultCode, totalCount, 이 페이지 item 수) 반환."""
    params = {
        "ServiceKey": SERVICE_KEY,
        "numOfRows": NUM_ROWS,
        "pageNo": page,
        "searchStDt": st,
        "searchEdDt": ed,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text

    root = ET.fromstring(text)                                   # XML 파싱 (읽기만, 값은 안 바꿈)
    code = root.findtext("header/resultCode")
    if code != "00":
        raise ValueError(f"resultCode={code} msg={root.findtext('header/resultMsg')}")
    total = int(root.findtext("body/totalCount") or 0)
    n_items = len(root.findall("body/items/item"))
    return text, code, total, n_items


def main():
    index_rows = []
    for ym in MONTHS:
        st, ed = month_range(ym)
        page, total = 1, None
        while True:
            text, code, total, n_items = fetch_page(st, ed, page)

            fname = f"fire_{ym}_p{page:02d}.xml"
            path = os.path.join(SAVE_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

            # 받는 즉시 인덱스 한 행
            index_rows.append({
                "file": fname,
                "source": BASE_URL,
                "params": f"numOfRows={NUM_ROWS}&pageNo={page}&searchStDt={st}&searchEdDt={ed}",
                "collected_at": datetime.now(),
                "size_bytes": len(text.encode("utf-8")),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "result_code": code,
                "total_count": total,
                "n_items": n_items,
            })

            n_pages = max(1, math.ceil(total / NUM_ROWS))
            if page >= n_pages:
                break
            page += 1
            time.sleep(0.3)

        print(f"{ym}: totalCount={total}, pages={page}")
        time.sleep(0.3)

    idx = pd.DataFrame(index_rows)
    idx.to_parquet(INDEX_PATH, index=False)
    idx.to_csv(INDEX_PATH.replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")

    print("\n=== 요약 ===")
    print(f"페이지(파일) 수: {len(idx)}")
    print(f"item 합계: {idx['n_items'].sum()}")
    by_month = idx.groupby(idx["file"].str[5:11])["total_count"].first()
    print(f"월별 totalCount 합계: {by_month.sum()}")
    print("item 합계 == totalCount 합계:", idx["n_items"].sum() == by_month.sum())


if __name__ == "__main__":
    main()