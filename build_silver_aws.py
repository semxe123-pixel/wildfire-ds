import os
import numpy as np
import pandas as pd

IN_DIR = "data/bronze/aws_raw"
OUT_DIR = "data/silver/aws"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

NUM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
RANGES = {"HM": (0, 100), "WS": (0, None), "WD": (0, 360), "TA": (-40, 45)}   # R4
TOTAL_HOURS = 17544


def apply_rules(raw):
    """Bronze 한 달 → Silver 한 달. (표, 규칙별 건수) 반환."""
    df = pd.DataFrame({
        "obs_time": pd.to_datetime(raw["YYMMDDHHMI"], format="%Y%m%d%H%M"),
        "STN": raw["STN"].astype(int),
        "src_file": raw["src_file"].astype(str),
    })
    for c in NUM:
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    n = {}

    # R1: 결측 표시값
    for c in NUM:
        m = df[c] <= -99
        n[f"R1_{c}"] = int(m.sum())
        df.loc[m, c] = np.nan

    # R2: 음수 강수
    df["qc_neg_rain"] = False
    for c in ["RN_HR1", "RN_DAY"]:
        m = df[c] < 0
        n[f"R2_{c}"] = int(m.sum())
        df.loc[m, "qc_neg_rain"] = True
        df.loc[m, c] = np.nan

    # R3: 기압 오류
    df["qc_pa_invalid"] = False
    for c in ["PA", "PS"]:
        m = df[c] < 500
        n[f"R3_{c}"] = int(m.sum())
        df.loc[m, "qc_pa_invalid"] = True
        df.loc[m, c] = np.nan

    # R4: 물리 범위
    df["qc_range"] = False
    for c, (lo, hi) in RANGES.items():
        m = df[c] < lo
        if hi is not None:
            m = m | (df[c] > hi)
        n[f"R4_{c}"] = int(m.sum())
        df.loc[m, "qc_range"] = True
        df.loc[m, c] = np.nan

    # R7: 유일성 (R5 전에 — 정렬·연속 계산이 중복에 민감)
    dup = int(df.duplicated(["STN", "obs_time"]).sum())
    n["R7_dup"] = dup
    if dup:
        raise ValueError(f"(STN, obs_time) 중복 {dup}행 — 확인 필요")

    # R5: TA 0.0 고착값 (같은 STN에서 3시간 이상 연속)
    df = df.sort_values(["STN", "obs_time"]).reset_index(drop=True)
    is_zero = df["TA"] == 0.0
    run_id = (is_zero != is_zero.groupby(df["STN"]).shift()).cumsum()
    run_len = is_zero.groupby(run_id).transform("sum")          # 0.0 구간이면 그 길이, 아니면 0
    m = is_zero & (run_len >= 3)
    n["R5_TA"] = int(m.sum())
    df["qc_stuck_zero"] = m
    df.loc[m, "TA"] = np.nan

    return df, n


def main():
    files = sorted(f for f in os.listdir(IN_DIR) if f.endswith(".parquet"))
    totals, per_stn = {}, pd.Series(dtype=int)
    rows = 0
    for f in files:
        raw = pd.read_parquet(os.path.join(IN_DIR, f))
        df, n = apply_rules(raw)
        ym = f.replace("aws_raw_", "").replace(".parquet", "")
        df.to_parquet(os.path.join(OUT_DIR, f"aws_silver_{ym}.parquet"), index=False)
        for k, v in n.items():
            totals[k] = totals.get(k, 0) + v
        per_stn = per_stn.add(df.groupby("STN").size(), fill_value=0)
        rows += len(df)
        print(f"{ym}: {len(df):,}행  R2 {n['R2_RN_HR1']+n['R2_RN_DAY']}  R3 {n['R3_PA']+n['R3_PS']}  "
              f"R4 {sum(v for k, v in n.items() if k.startswith('R4'))}  R5 {n['R5_TA']}")

    # R6: 관측소 결측률
    cov = pd.DataFrame({"n_rows": per_stn.astype(int)})
    cov["coverage"] = (cov["n_rows"] / TOTAL_HOURS).round(4)
    cov["low_coverage"] = cov["coverage"] < 0.5
    cov.index.name = "STN"
    cov.reset_index().to_parquet("data/silver/aws_station_coverage.parquet", index=False)
    cov.reset_index().to_csv("data/silver/aws_station_coverage.csv", index=False, encoding="utf-8-sig")

    # 보고서
    lines = ["# Silver 기상 적용 결과 (24개월)", "",
             f"- 입력 행: {rows:,} (Bronze 12,549,269와 일치: {rows == 12549269})",
             f"- 관측소 {len(cov)}곳, low_coverage(결측률>50%) {int(cov.low_coverage.sum())}곳: "
             f"{cov[cov.low_coverage].coverage.round(2).to_dict()}", "", "| 규칙 | 건수 |", "| --- | --- |"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(totals.items())]
    open("docs/silver_report_aws.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()