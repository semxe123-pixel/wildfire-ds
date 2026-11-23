"""Gold G4: 검증 보고 → data/gold/gold_report.md, data/gold/gold_feature_stats.csv (변수 정의서용 통계)

Gold Feature 표(G2)와 학습 표(G3)가 규칙대로 만들어졌는지 숫자로 확인한다. 표를 고치지 않음 (읽기만).

검사
  R1 행 수: 월마다 (후보 관측소 표의 행정동 × 유효 기간 안 시각 수)와 같은지
  R2 (행정동, 기준 시각) 중복 없음 · 기준 시각이 그 달 안·정시
  R3 열별 빈 값 비율
  R4 값 범위: 기온 -40~60, 습도 0~100, 풍속 0~75, 강수 0 이상, 실효습도 0~100, 건조 지속 0~120, 풍향 sin²+cos² = 1
  R5 추적: 직전 값을 이어 쓴 비율(age 1~3), 값 없음(age -1), 쓴 관측소 거리, 확인 필요 값을 쓴 비율
  R6 누출 다시 확인: 달마다 무작위 10줄의 기온을 aws_silver에서 'D4 + t-1시간' 규칙으로 다시 계산해 같은지
  R7 학습 표: 행 수, Feature 빈 값, 기준 시각 ≤ 발생 시각
  통계 표: 열마다 빈 값 비율·최소·1%·중앙값·99%·최대 (최소·최대는 전부, 분위수는 달마다 2% 표본)

입력  data/gold/feature/*.parquet, data/gold/gold_train.parquet, data/silver/stn_assign.parquet, data/silver/aws_silver_*.parquet
출력  data/gold/gold_report.md, data/gold/gold_feature_stats.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ = "Asia/Seoul"
G = "data/gold"
lines, summary = [], []
RANGE = {"ta": (-40, 60), "ta_max_24h": (-40, 60), "ta_min_24h": (-40, 60), "hm": (0, 100), "hm_min_24h": (0, 100),
         "hm_mean_24h": (0, 100), "eff_hm": (0, 100), "ws": (0, 75), "ws_max_24h": (0, 75), "ws_mean_6h": (0, 75),
         "rn_1h": (0, 500), "rn_day": (0, 2000), "rn_sum_24h": (0, 2000), "rn_sum_72h": (0, 3000), "dry_hours": (0, 120)}


def md(s=""):
    lines.append(s)
    print(s)


def check(code, name, ok, detail, warn=False):
    mark = "통과" if ok else ("주의" if warn else "실패")
    summary.append((code, name, mark, detail))
    md(f"- **[{mark}] {code} {name}**: {detail}")


def table(df):
    try:
        md(df.to_markdown(index=False))
    except ImportError:
        md("```\n" + df.to_string(index=False) + "\n```")
    md()


md("# Gold 검증 보고")
md(f"\n생성: {pd.Timestamp.now(tz=TZ):%Y-%m-%d %H:%M} KST\n")
files = sorted(glob.glob(f"{G}/feature/gold_feature_*.parquet"))
assign = pd.read_parquet("data/silver/stn_assign.parquet",
                         columns=["admin_dong_code", "valid_from", "valid_to", "month", "element", "rank", "station_id"])
rows, na_sum, n_total, bad_rng, dup, out_m = [], None, 0, {}, 0, 0
mins, maxs, samples, track = {}, {}, [], []
for f in files:
    m = os.path.basename(f)[13:19]
    d = pd.read_parquet(f)
    ms = pd.Timestamp(f"{m[:4]}-{m[4:]}-01", tz=TZ)
    me = ms + pd.offsets.MonthBegin(1)
    gr = assign[assign.month == m][["admin_dong_code", "valid_from", "valid_to"]].drop_duplicates()
    exp = int(((gr.valid_to.clip(upper=me) - gr.valid_from.clip(lower=ms)) / pd.Timedelta(hours=1)).sum())
    rows.append({"월": m, "행": len(d), "기대 행": exp, "행정동": d.admin_dong_code.nunique()})
    dup += int(d.duplicated(["admin_dong_code", "forecast_reference_time"]).sum())
    t = d.forecast_reference_time
    out_m += int(((t < ms) | (t >= me) | (t != t.dt.floor("h"))).sum())
    num = d.select_dtypes("number")
    na = num.isna().sum()
    na_sum = na if na_sum is None else na_sum + na
    n_total += len(d)
    for c in num:
        x = pd.to_numeric(num[c], errors="coerce")
        mins[c] = min(mins.get(c, np.inf), float(x.min())) if x.notna().any() else mins.get(c, np.nan)
        maxs[c] = max(maxs.get(c, -np.inf), float(x.max())) if x.notna().any() else maxs.get(c, np.nan)
    for c, (lo, hi) in RANGE.items():
        if c in d:
            bad_rng[c] = bad_rng.get(c, 0) + int(((d[c] < lo) | (d[c] > hi)).sum())
    unit = (d.wd_sin ** 2 + d.wd_cos ** 2).dropna()
    bad_rng["wd_sin²+cos²"] = bad_rng.get("wd_sin²+cos²", 0) + int((unit.sub(1).abs() > 1e-3).sum())
    samples.append(d.sample(frac=0.02, random_state=0))
    tr = {"월": m}
    for e in ("ta", "hm", "ws", "rn_hr1", "pa"):
        a = d[f"{e}_age_h"]
        tr[f"{e} 이어씀%"] = round(float(a.between(1, 3).mean() * 100), 3)
        tr[f"{e} 없음%"] = round(float((a == -1).mean() * 100), 3)
    tr["확인필요 사용%"] = round(float((d.n_check_used > 0).mean() * 100), 3)
    track.append(tr)
    print(f"  읽음 {m}", flush=True)

md("## R1·R2 행 수와 키\n")
rw = pd.DataFrame(rows)
table(rw)
check("R1", "월마다 행 수 = 기대 행 수", bool((rw["행"] == rw["기대 행"]).all()),
      f"전체 {rw['행'].sum():,}행 / 기대 {rw['기대 행'].sum():,}")
check("R2", "(행정동, 기준 시각) 중복 없음 · 그 달 안 정시", dup == 0 and out_m == 0, f"중복 {dup} / 범위 밖 {out_m}")

md("\n## R3·R4 빈 값과 값 범위\n")
na_pct = (na_sum / n_total * 100).round(3)
check("R3", "현재 값(ta, hm, ws, rn_1h, pa, ps) 빈 값 1% 미만", bool((na_pct[["ta", "hm", "ws", "rn_1h", "pa", "ps"]] < 1).all()),
      ", ".join(f"{c} {na_pct[c]}%" for c in ["ta", "hm", "ws", "rn_1h", "pa", "ps", "wd_sin", "eff_hm", "rn_sum_72h"]),
      warn=True)
check("R4", "값 범위", sum(bad_rng.values()) == 0, ", ".join(f"{k} {v}" for k, v in bad_rng.items() if v) or "벗어난 값 없음")

md("\n## R5 추적 (월별)\n")
table(pd.DataFrame(track))

# 통계 표 (변수 정의서용)
smp = pd.concat(samples, ignore_index=True)
st = []
for c in na_sum.index:
    x = pd.to_numeric(smp[c], errors="coerce").dropna()
    st.append({"열": c, "빈 값%": na_pct[c], "최소": mins.get(c), "1%": x.quantile(.01) if len(x) else np.nan,
               "중앙값": x.median() if len(x) else np.nan, "99%": x.quantile(.99) if len(x) else np.nan, "최대": maxs.get(c)})
st = pd.DataFrame(st)
num_cols = ["최소", "1%", "중앙값", "99%", "최대"]
st[num_cols] = st[num_cols].astype(float).round(3)
st.to_csv(f"{G}/gold_feature_stats.csv", index=False, encoding="utf-8-sig")
md("## 열별 통계 (분위수는 2% 표본)\n")
table(st)

# R6 누출 다시 확인
md("## R6 누출 다시 확인 (기온)\n")
okc, n6 = 0, 0
for f in files:
    m = os.path.basename(f)[13:19]
    d = pd.read_parquet(f, columns=["admin_dong_code", "forecast_reference_time", "ta", "ta_age_h"])
    d = d[d.ta_age_h == 0].sample(10, random_state=1)
    ms = pd.Timestamp(f"{m[:4]}-{m[4:]}-01", tz=TZ)
    prev = (ms - pd.Timedelta(days=1)).strftime("%Y%m")
    aw = pd.concat([pd.read_parquet(x, columns=["observed_at", "station_id", "TA", "qc_TA"])
                    for x in (f"data/silver/aws_silver_{prev}.parquet", f"data/silver/aws_silver_{m}.parquet") if os.path.exists(x)])
    aw = aw[aw.qc_TA.isin([0, 2]) & aw.TA.notna()].set_index(["station_id", "observed_at"]).TA
    a = assign[(assign.month == m) & (assign.element == "TA")]
    for r in d.itertuples():
        t = r.forecast_reference_time
        c = a[(a.admin_dong_code == r.admin_dong_code) & (a.valid_from <= t) & (a.valid_to > t)].sort_values("rank")
        val = next((aw[(s, t - pd.Timedelta(hours=1))] for s in c.station_id if (s, t - pd.Timedelta(hours=1)) in aw.index), np.nan)
        n6 += 1
        okc += bool(np.isclose(val, r.ta))
check("R6", "무작위 줄의 기온 = 't-1시간에 값이 있는 가장 가까운 후보'의 값", okc == n6, f"{okc}/{n6}")

# R7 학습 표
md("\n## R7 학습 표 gold_train\n")
if os.path.exists(f"{G}/gold_train.parquet"):
    tr = pd.read_parquet(f"{G}/gold_train.parquet")
    feats = ["ta", "hm", "ws", "pa", "ps", "rn_1h", "rn_day", "ta_max_24h", "ta_min_24h", "hm_min_24h", "hm_mean_24h",
             "ws_max_24h", "ws_mean_6h", "rn_sum_24h", "rn_sum_72h", "dry_hours", "eff_hm", "wind_u", "wind_v"]
    nf = tr[feats].isna().sum()
    check("R7-1", "학습 표 행 수 (행정동을 정한 사건 수)", len(tr) == tr.fire_id.nunique(), f"{len(tr)}행 · {tr.split.value_counts().to_dict()}")
    check("R7-2", "Feature 빈 값 없음 (풍향 sin·cos 제외)", int(nf.sum()) == 0, nf[nf > 0].to_dict() or "없음", warn=True)
    check("R7-3", "기준 시각 ≤ 발생 시각", bool((tr.forecast_reference_time <= tr.ignition_at).all()), "발생 이후 정보 없음")
else:
    check("R7", "학습 표", False, "gold_train.parquet 없음 (build_gold_train.py 먼저)")

md("\n## 요약\n")
table(pd.DataFrame(summary, columns=["번호", "검사", "결과", "내용"]))
with open(f"{G}/gold_report.md", "w", encoding="utf-8") as fo:
    fo.write("\n".join(lines) + "\n")
print(f"[저장] {G}/gold_report.md, {G}/gold_feature_stats.csv")