import os
import xml.etree.ElementTree as ET
import pandas as pd

RAW_DIR = "data/bronze/fire"
INDEX_PATH = "data/bronze/fire_index.parquet"
OUT_PATH = "data/bronze/fire_raw.parquet"

# 응답 <item> 안의 태그 이름 그대로. 원천 순서(알파벳)대로 둔다.
FIELDS = [
    "damagearea", "endday", "endmonth", "endtime", "endyear", "firecause",
    "locbunji", "locdong", "locgungu", "locmenu", "locsi",
    "startday", "startdayofweek", "startmonth", "starttime", "startyear",
]


def parse_file(path):
    name = os.path.basename(path)
    root = ET.parse(path).getroot()
    rows = []
    for item in root.findall("body/items/item"):
        row = {}
        for f in FIELDS:
            el = item.find(f)
            row[f] = el.text if (el is not None and el.text is not None) else ""   # 태그 없으면 빈 문자열
        row["src_file"] = name
        rows.append(row)
    return rows


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".xml"))
    rows = []
    for name in files:
        rows.extend(parse_file(os.path.join(RAW_DIR, name)))

    df = pd.DataFrame(rows, columns=FIELDS + ["src_file"], dtype="string")
    df.to_parquet(OUT_PATH, index=False)

    idx = pd.read_parquet(INDEX_PATH)
    expected = int(idx["n_items"].sum())
    print(f"파일 {len(files)}개 → {len(df)}행")
    print(f"인덱스 n_items 합계: {expected}")
    print("일치" if len(df) == expected else "불일치 — 확인 필요")

    # 원천 특성 미리보기 (값은 바꾸지 않고 보기만)
    print("\nlocmenu 빈 값(동 지역):", (df["locmenu"] == "").sum())
    print("locsi 종류:", sorted(df["locsi"].unique().tolist()))
    print("firecause 상위 5:\n", df["firecause"].value_counts().head())


if __name__ == "__main__":
    main()