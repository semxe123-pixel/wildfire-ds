"""기상 탐색 공통: Bronze 24개월을 한 표로 읽는다. 저장하지 않는다.

- 시각은 한국 시간(Asia/Seoul)으로 바꾼다 (공통 기준)
- 기상청 결측 표시(-99 이하)는 빈 값(NaN)으로 바꾼다. 원천이 정한 표시라서 탐색 단계부터 적용
- 메모리를 줄이려고 숫자는 float32로 읽는다
"""
import os
import numpy as np
import pandas as pd

IN_DIR = "data/bronze/aws_raw"
NUM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
TZ = "Asia/Seoul"


def load_aws():
    frames = []
    for f in sorted(os.listdir(IN_DIR)):
        if not f.endswith(".parquet"):
            continue
        raw = pd.read_parquet(os.path.join(IN_DIR, f))
        df = pd.DataFrame({
            "obs_time": pd.to_datetime(raw["YYMMDDHHMI"], format="%Y%m%d%H%M").dt.tz_localize(TZ),
            "STN": raw["STN"].astype("int32"),
        })
        for c in NUM:
            v = pd.to_numeric(raw[c]).astype("float32")
            df[c] = v.where(v > -99)          # -99 이하 → NaN
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["STN", "obs_time"]).reset_index(drop=True)


def load_station_height():
    """관측소 × 월의 높이(HT, m). 매달 1일 지점정보를 그 달에 씀"""
    stn = pd.read_parquet("data/bronze/stn_raw.parquet")
    stn = stn[stn["src_file"].str.contains("010000")]          # 2025-12-31 23시 파일은 빼고 매달 1일만
    return pd.DataFrame({
        "STN": stn["STN_ID"].astype("int32"),
        "month": stn["src_file"].str[8:14],                     # stn_aws_YYYYMM010000.txt → YYYYMM
        "HT": pd.to_numeric(stn["HT"], errors="coerce").astype("float32"),
    })


def q(s, ps=(0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1)):
    """분위수 표 한 줄: 0%(최솟값) ~ 100%(최댓값)"""
    s = s.dropna()
    return {f"{p*100:g}%": round(float(s.quantile(p)), 2) for p in ps} | {"n": len(s)}


def load_station_info():
    """관측소 × 월의 이름·좌표·높이. 그 달 1일 지점정보를 쓰고,
    그 달 1일에 없는 관측소(월 중간 신설)는 처음 나타나는 다음 파일의 값을 쓴다."""
    stn = pd.read_parquet("data/bronze/stn_raw.parquet")
    info = pd.DataFrame({
        "STN": stn["STN_ID"].astype("int32"),
        "snap": stn["src_file"].str[8:20],                       # YYYYMMDDHHMI
        "STN_KO": stn["STN_KO"].str.strip(),
        "LON": pd.to_numeric(stn["LON"], errors="coerce"),
        "LAT": pd.to_numeric(stn["LAT"], errors="coerce"),
        "HT": pd.to_numeric(stn["HT"], errors="coerce"),
    })
    months = pd.period_range("2024-01", "2025-12", freq="M").strftime("%Y%m")
    rows = []
    for s, g in info.sort_values("snap").groupby("STN"):
        for mo in months:
            cand = g[g.snap >= mo + "010000"]                    # 그 달 1일 또는 그 뒤의 첫 파일
            if len(cand):
                rows.append(cand.iloc[0].to_dict() | {"month": mo})
    return pd.DataFrame(rows).drop(columns="snap")