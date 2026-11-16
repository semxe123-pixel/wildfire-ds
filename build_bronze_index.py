import os
import hashlib
from datetime import datetime
import pandas as pd

RAW_DIR = "data/bronze/aws"
OUT_PARQUET = "data/bronze/aws_index.parquet"
OUT_CSV = "data/bronze/aws_index.csv"
SOURCE = "apihub.kma.go.kr/api/typ01/url/awsh.php"


def sha256_of(path):
    """파일 내용의 SHA-256 checksum을 16진수 문자열로 돌려준다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:              # 'rb' = 바이트로 읽기. checksum은 바이트 기준
        for chunk in iter(lambda: f.read(1 << 20), b""):   # 1MB씩 읽어 메모리 절약
            h.update(chunk)
    return h.hexdigest()


def inspect_file(path):
    """파일 하나를 읽어 인덱스 한 행(딕셔너리)을 만든다."""
    name = os.path.basename(path)
    tm = name.replace(".txt", "")

    n_rows = 0
    has_end = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "#7777END":
                has_end = True
            elif s and not s.startswith("#"):   # 비어 있지 않고 #으로 시작하지 않으면 데이터 행
                n_rows += 1

    stat = os.stat(path)
    return {
        "file": name,
        "tm": tm,
        "source": SOURCE,
        "params": f"tm={tm}&help=0",
        "collected_at": datetime.fromtimestamp(stat.st_mtime),
        "size_bytes": stat.st_size,
        "sha256": sha256_of(path),
        "n_rows": n_rows,
        "has_end": has_end,
    }


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    print(f"파일 {len(files)}개 검사 시작")

    rows = []
    for i, name in enumerate(files, 1):
        rows.append(inspect_file(os.path.join(RAW_DIR, name)))
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 요약 출력
    print("\n=== 인덱스 요약 ===")
    print(f"행 수(파일 수): {len(df)}")
    print(f"데이터 행 합계: {df['n_rows'].sum():,}")
    print(f"has_end False 개수: {(~df['has_end']).sum()}")
    print(f"n_rows 최소/중앙/최대: {df['n_rows'].min()} / {df['n_rows'].median()} / {df['n_rows'].max()}")
    print(f"수집 기간: {df['collected_at'].min()} ~ {df['collected_at'].max()}")
    print(f"tm 범위: {df['tm'].min()} ~ {df['tm'].max()}")
    print(f"tm 중복: {df['tm'].duplicated().sum()}")


if __name__ == "__main__":
    main()