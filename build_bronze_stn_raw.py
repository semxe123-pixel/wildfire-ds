import os
import re
import hashlib
from datetime import datetime
import pandas as pd

RAW_DIR = "data/bronze/stn"
COLUMNS = ["STN_ID", "LON", "LAT", "STN_SP", "HT", "HT_WD", "LAU_ID", "STN_AD",
           "STN_KO", "STN_EN", "FCT_ID", "LAW_ID", "BASIN", "LAW_ADDR"]

# 한 줄을 14개 열로 자르는 규칙. 값은 그대로, 자르는 위치만 정한다.
LINE_RE = re.compile(
    r"^\s*(?P<STN_ID>\d+)\s+(?P<LON>[\d.]+)\s+(?P<LAT>[\d.]+)\s+(?P<STN_SP>\S+)\s+"
    r"(?P<HT>\S+)\s+(?P<HT_WD>\S+)\s+(?P<LAU_ID>\S+)\s+(?P<STN_AD>\S+)\s+"
    r"(?P<STN_KO>.+?)\s{2,}(?P<STN_EN>.+?)\s+"          # 이름 칸은 폭 20, 칸 사이는 공백 2개 이상
    r"(?P<FCT_ID>\S{8})\s+(?P<LAW_ID>\S{10})\s+"
    r"(?P<BASIN>\d+|-+)\s*(?P<LAW_ADDR>.*?)\s*$"          # BASIN은 숫자 또는 ----. 뒤의 공백은 실제로 1칸(전 행)이고, \s*라서 0칸이어도 잘림
)


def parse_file(path):
    name = os.path.basename(path)
    rows, bad = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            m = LINE_RE.match(s)
            if m:
                rows.append({**m.groupdict(), "src_file": name})
            else:
                bad.append((name, s))
    return rows, bad


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    frames, index_rows, all_bad = [], [], []
    for name in files:
        path = os.path.join(RAW_DIR, name)
        rows, bad = parse_file(path)
        all_bad.extend(bad)
        df = pd.DataFrame(rows, columns=COLUMNS + ["src_file"], dtype="string")
        frames.append(df)

        with open(path, "rb") as f:
            data = f.read()
        tm = name.replace("stn_aws_", "").replace(".txt", "")
        index_rows.append({
            "file": name, "source": "apihub.kma.go.kr/api/typ01/url/stn_inf.php",
            "params": f"inf=AWS&stn=&tm={tm}&help=1",
            "collected_at": datetime.fromtimestamp(os.stat(path).st_mtime),
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "n_rows": len(rows) + len(bad), "n_parsed": len(rows), "n_bad": len(bad),
        })
        print(f"{name}: 파싱 {len(rows)}행, 실패 {len(bad)}행")

    pd.DataFrame(index_rows).to_parquet("data/bronze/stn_index.parquet", index=False)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet("data/bronze/stn_raw.parquet", index=False)

    if all_bad:
        print("\n파싱 실패 예시 (최대 5):")
        for name, s in all_bad[:5]:
            print(f"  [{name}] {s}")

    a = set(frames[0]["STN_ID"]); b = set(frames[1]["STN_ID"])
    print(f"\n2024초 {len(a)} / 2025말 {len(b)} / 합집합 {len(a | b)}")
    print(f"2024초에만(폐지?) {len(a - b)} / 2025말에만(신설?) {len(b - a)}")
    print("\n미리보기:")
    print(raw[["STN_ID", "STN_KO", "LAW_ID", "BASIN", "LAW_ADDR"]].head(8).to_string())
    print("\nLAW_ID 비어있거나 ---- 인 행:", (~raw["LAW_ID"].str.fullmatch(r"\d{10}")).sum())


if __name__ == "__main__":
    main()