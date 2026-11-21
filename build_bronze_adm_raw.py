import os
import glob
import hashlib
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook

ADM_DIR = "data/bronze/adm"
CHECK_CSV = "data/bronze/adm_raw_check.csv"
POST_URL = {
    "20260101": "https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardList.do?bbsId=BBSMSTR_000000000052 (2026.1.1. 시행 게시물)",
    "20261026": "같은 게시판 (2026.10.30. 기준 알림 게시물)",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_excel_rows(path):
    """엑셀을 pandas와 다른 방법(openpyxl)으로 열어 (시트 수, 데이터 행 수)를 센다.
    데이터 행 수 = 값이 하나라도 있는 행 수 - 1 (첫 줄은 열 이름)."""
    wb = load_workbook(path, read_only=True)
    ws = wb.worksheets[0]
    n_nonempty = sum(
        1 for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)
    )
    n_sheets = len(wb.worksheets)
    wb.close()
    return n_sheets, n_nonempty - 1


def main():
    # 1) 원본(zip, Layout) 인덱스
    originals = [f for f in os.listdir(ADM_DIR) if f.endswith((".zip", ".hwpx"))]
    idx = []
    for name in originals:
        p = os.path.join(ADM_DIR, name)
        date = "20261026" if "20261026" in name else "20260101"
        idx.append({
            "file": name, "source": POST_URL[date],
            "collected_at": datetime.fromtimestamp(os.stat(p).st_mtime),
            "size_bytes": os.stat(p).st_size, "sha256": sha256_of(p),
        })
    pd.DataFrame(idx).to_parquet(os.path.join(ADM_DIR, "..", "adm_index.parquet"), index=False)
    print("원본 인덱스:", len(idx), "개")

    # 2) 구조화: 세 폴더의 xlsx 9개 → 문자열 그대로
    frames = []
    checks = []                                        # 건수 일치 기록 (엑셀 1개 = 1행)
    for folder in sorted(d for d in os.listdir(ADM_DIR) if os.path.isdir(os.path.join(ADM_DIR, d))):
        for x in sorted(glob.glob(os.path.join(ADM_DIR, folder, "*.xlsx"))):
            base = os.path.basename(x)
            kind = base.split(".")[0]                  # KIKmix / KIKcd_H / KIKcd_B
            df = pd.read_excel(x, dtype=str)

            # 건수 일치: 다른 방법으로 센 엑셀 행 수와 읽은 행 수를 비교
            n_sheets, n_excel = count_excel_rows(x)
            ok = (len(df) == n_excel) and (n_sheets == 1)
            checks.append({
                "src_file": f"{folder}/{base}", "table": kind, "snapshot": folder,
                "n_sheets": n_sheets, "n_excel_rows": n_excel, "n_read_rows": len(df),
                "match": ok,
            })

            df["table"] = kind
            df["snapshot"] = folder                    # jscode20260101 / ..._malso / jscode_latest_malso
            df["src_file"] = f"{folder}/{base}"
            frames.append(df)
            mark = "일치" if ok else "불일치 — 확인 필요"
            print(f"  {folder:24s} {kind:8s} 엑셀 {n_excel:>8,}행 / 읽은 행 {len(df):>8,} / 시트 {n_sheets} → {mark}")

    raw = pd.concat(frames, ignore_index=True).astype("string")
    raw.to_parquet(os.path.join(ADM_DIR, "..", "adm_raw.parquet"), index=False)
    print(f"\n구조화 표: {len(raw):,}행 → data/bronze/adm_raw.parquet")

    # 3) 건수 일치 검증
    chk = pd.DataFrame(checks)
    chk.to_csv(CHECK_CSV, index=False, encoding="utf-8-sig")
    expected = int(chk["n_excel_rows"].sum())
    print("\n=== 건수 검증 ===")
    print(f"엑셀 행 수 합계   : {expected:,}")
    print(f"구조화 표 행 수   : {len(raw):,}")
    print(f"불일치 엑셀 수    : {(~chk['match']).sum()}")
    print("일치" if (len(raw) == expected and chk["match"].all()) else "불일치 — 확인 필요")
    print(f"기록: {CHECK_CSV}")

    # 4) 미리보기: KIKmix 스냅샷별 말소일자 상태
    mix = raw[raw["table"] == "KIKmix"]
    for snap, g in mix.groupby("snapshot"):
        has_end = g["말소일자"].notna().sum()
        print(f"\n[{snap}] KIKmix {len(g):,}행, 말소일자 있음 {has_end:,}행")
        print("   생성일자 범위:", g["생성일자"].min(), "~", g["생성일자"].max())
        if has_end:
            print("   말소일자 범위:", g["말소일자"].min(), "~", g["말소일자"].max())
            print("   2024~2025 사이 말소:", g["말소일자"].between("20240101", "20251231").sum(), "행")
            print("   2026-07-01 이후 생성:", (g["생성일자"] >= "20260701").sum(), "행")


if __name__ == "__main__":
    main()