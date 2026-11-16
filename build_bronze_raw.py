import os
from collections import defaultdict
import pandas as pd

RAW_DIR = "data/bronze/aws"
OUT_DIR = "data/bronze/aws_raw"
INDEX_PATH = "data/bronze/aws_index.parquet"
os.makedirs(OUT_DIR, exist_ok=True)

# 파일 헤더(# YYMMDDHHMI STN TA ...)와 같은 순서. 원천 이름 그대로.
COLUMNS = ["YYMMDDHHMI", "STN", "TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]


def parse_file(path):
    """txt 하나 → (정상 행 리스트, 이상 행 리스트).
    정상 행 = 토큰 10개 + src_file. 값은 전부 문자열."""
    name = os.path.basename(path)
    rows, bad = [], []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tokens = s.split()                      # 공백(여러 개여도) 기준으로 쪼갬
            if len(tokens) == len(COLUMNS):
                rows.append(tokens + [name])
            else:
                bad.append({"src_file": name, "lineno": lineno, "n_tokens": len(tokens), "line": s})
    return rows, bad


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))

    # 월별로 묶기: '202401010000.txt' → '202401'
    by_month = defaultdict(list)
    for name in files:
        by_month[name[:6]].append(name)
    print(f"파일 {len(files)}개, {len(by_month)}개월")

    total_rows = 0
    all_bad = []
    for month, names in sorted(by_month.items()):
        rows = []
        for name in names:
            r, b = parse_file(os.path.join(RAW_DIR, name))
            rows.extend(r)
            all_bad.extend(b)

        df = pd.DataFrame(rows, columns=COLUMNS + ["src_file"], dtype="string")
        out = os.path.join(OUT_DIR, f"aws_raw_{month}.parquet")
        df.to_parquet(out, index=False)
        total_rows += len(df)
        print(f"  {month}: 파일 {len(names)}개 → {len(df):,}행")

    # 이상 행 기록
    if all_bad:
        pd.DataFrame(all_bad).to_csv("data/bronze/aws_raw_badlines.csv", index=False, encoding="utf-8-sig")

    # 건수 일치 검증 (Step 3)
    idx = pd.read_parquet(INDEX_PATH)
    expected = int(idx["n_rows"].sum())
    print("\n=== 건수 검증 ===")
    print(f"인덱스 n_rows 합계 : {expected:,}")
    print(f"구조화 표 행 수     : {total_rows:,}")
    print(f"이상 행(토큰≠10)   : {len(all_bad):,}")
    print("일치" if total_rows + len(all_bad) == expected else "불일치 — 확인 필요")


if __name__ == "__main__":
    main()