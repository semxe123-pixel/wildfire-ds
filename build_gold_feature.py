"""Gold G2: 행정동 × 1시간 Feature 표 → data/gold/feature/gold_feature_YYYYMM.parquet (24개)

한 줄 = 행정동 하나 × 기준 시각(forecast_reference_time) 하나.
"그 시각에 이미 받을 수 있었던 관측"만으로 만든 날씨 Feature. 기간 2024-01-01 00시 ~ 2025-12-31 23시, 그 시각에 유효한 행정동 전부.

규칙 (G1)
  F1 기준 시각 t의 값 = 관측 시각 t-1시간의 관측 (available_at = 관측 + 1시간 ≤ t). t 이후 관측은 절대 안 씀
  F2 행정동 값 = 후보 관측소(stn_assign, 30 km 안 최대 5개)를 가까운 순으로 보고, 그 시각 값이 있는 첫 관측소 값 (D4)
     값이 있다 = 판정 0(정상) 또는 2(확인 필요). 1(결측)·3(제거)은 빈 값이라 다음 후보로 넘어감
  F3 후보 5곳 모두 값이 없으면 직전 값을 최대 3시간까지 이어 씀 (기온·습도·풍속·풍향·기압). 강수는 이어 쓰지 않음
     앞뒤 보간은 안 함 (뒤 값을 쓰면 t에 몰랐던 정보가 섞임). 이어 쓴 시간은 {요소}_age_h에 남김
  F4 풍향은 풍속 0.2 m/s 미만이면 빈 값 (WMO: 무풍에서는 방향이 의미 없음). 각도 대신 sin·cos 두 열로 저장
     (359°와 1°가 숫자로는 멀지만 실제로는 가까우므로)
  F5 누적 Feature는 t-1시간까지의 지난 W시간 값으로 계산. 창 안에 값이 75% 이상 있을 때만 계산, 아니면 빈 값
  F6 앞을 보는 창(최대 120시간)도 그 기준 시각의 후보 관측소로 계산 (행정동 코드가 바뀐 직후에도 같은 땅의 지난 날씨가 이어짐)
  F7 실효습도: He = (1-r)(H0 + r·H1 + r²·H2 + r³·H3 + r⁴·H4), r = 0.7, Hk = k일 전 24시간 평균 습도
     (기상청 건조특보 기준에 쓰는 값. 달력 날짜 대신 't-1시간부터 24시간씩'으로 계산 → 기준 시각마다 다시 계산 가능)
  F8 고도 보정은 하지 않음 (행정동 높이 자료 없음). 대신 쓴 관측소까지 거리(km)를 남김

입력  data/silver/aws_silver_YYYYMM.parquet, data/silver/stn_assign.parquet
출력  data/gold/feature/gold_feature_YYYYMM.parquet
실행  python build_gold_feature.py            (24개월 전부)
      python build_gold_feature.py 202503     (한 달만, 시험용)
"""
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)          # 전부 빈 창의 nanmax 등
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ = "Asia/Seoul"
S, OUTDIR = "data/silver", "data/gold/feature"
ELEM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
CARRY = {"TA", "WD", "WS", "HM", "PA", "PS"}                         # F3
CARRY_H = 3
LB = 120                                                            # F6: 앞을 보는 최대 시간
MIN_COVER = 0.75                                                    # F5
CALM = 0.2                                                          # F4
R_EFF = 0.7                                                         # F7
MONTHS = [f"{y}{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def month_start(m):
    return pd.Timestamp(f"{m[:4]}-{m[4:]}-01", tz=TZ)


def load_aws(months):
    parts = []
    for m in months:
        f = f"{S}/aws_silver_{m}.parquet"
        if os.path.exists(f):
            parts.append(pd.read_parquet(f, columns=["observed_at", "station_id"] + ELEM + [f"qc_{e}" for e in ELEM]))
    return pd.concat(parts, ignore_index=True) if parts else None


def window_stat(arr, end_idx, w, how):
    """arr (G, Ho)에서 각 기준 시각의 창 [end-w+1, end]의 통계. 창 안 값이 MIN_COVER 미만이면 NaN
    합·평균은 누적합으로(메모리 절약), 최대·최소는 창을 펼쳐서 계산"""
    G, Ho = arr.shape
    out = np.full((G, len(end_idx)), np.nan, dtype=np.float32)
    ok = end_idx - w + 1 >= 0
    if not ok.any():
        return out
    e = end_idx[ok]
    isv = ~np.isnan(arr)
    ccnt = np.concatenate([np.zeros((G, 1)), np.cumsum(isv, axis=1)], axis=1)
    cnt = ccnt[:, e + 1] - ccnt[:, e + 1 - w]
    if how in ("sum", "mean"):
        csum = np.concatenate([np.zeros((G, 1)), np.cumsum(np.where(isv, arr, 0), axis=1, dtype=np.float64)], axis=1)
        v = csum[:, e + 1] - csum[:, e + 1 - w]
        if how == "mean":
            v = v / np.maximum(cnt, 1)
    else:
        win = np.lib.stride_tricks.sliding_window_view(arr, w, axis=1)[:, e - w + 1, :]   # (G, H_ok, w)
        v = np.nanmax(win, axis=2) if how == "max" else np.nanmin(win, axis=2)
    out[:, ok] = np.where(cnt >= MIN_COVER * w, v, np.nan)
    return out


def build_month(m, assign):
    t0 = time.time()
    ms = month_start(m)
    me = ms + pd.offsets.MonthBegin(1)
    ref = pd.date_range(ms, me - pd.Timedelta(hours=1), freq="h")            # 기준 시각
    obs = pd.date_range(ms - pd.Timedelta(hours=LB), me - pd.Timedelta(hours=2), freq="h")   # 관측 시각 축
    H, Ho = len(ref), len(obs)
    end_idx = np.arange(H) + LB - 1                                          # 기준 시각 t → 관측 t-1시간의 위치 (F1)

    a = assign[assign.month == m]
    groups = a[["admin_dong_code", "valid_from", "valid_to"]].drop_duplicates().reset_index(drop=True)
    G = len(groups)
    gkey = {k: i for i, k in enumerate(map(tuple, groups.values))}

    prev = (ms - pd.Timedelta(days=1)).strftime("%Y%m")
    aws = load_aws([prev, m])
    aws = aws[(aws.observed_at >= obs[0]) & (aws.observed_at <= obs[-1])]
    stations = sorted(set(aws.station_id) | set(a.station_id))
    sidx = {s: i for i, s in enumerate(stations)}
    ri = aws.station_id.map(sidx).to_numpy()
    ci = ((aws.observed_at - obs[0]) / pd.Timedelta(hours=1)).astype(int).to_numpy()

    # 요소별 관측소 × 관측 시각 행렬 (값이 있는 것 = 판정 0·2)
    V, CHK = {}, {}
    for e in ELEM:
        v = np.full((len(stations), Ho), np.nan, dtype=np.float32)
        q = aws[f"qc_{e}"].to_numpy()
        good = np.isin(q, [0, 2]) & aws[e].notna().to_numpy()
        v[ri[good], ci[good]] = aws[e].to_numpy(dtype=np.float32)[good]
        c = np.zeros((len(stations), Ho), dtype=bool)
        c[ri[q == 2], ci[q == 2]] = True
        V[e], CHK[e] = v, c

    out = {}
    track_dist, track_age, chk_used = {}, {}, np.zeros((G, H), dtype=np.int8)
    ws_at_wd = None
    for e in ELEM:
        ae = a[a.element == e]
        cidx = np.full((G, 5), -1, dtype=np.int64)
        cdist = np.full((G, 5), np.nan, dtype=np.float32)
        g_of = np.array([gkey[k] for k in map(tuple, ae[["admin_dong_code", "valid_from", "valid_to"]].values)])
        r_of = ae["rank"].to_numpy().astype(int) - 1
        cidx[g_of, r_of] = ae.station_id.map(sidx).to_numpy()
        cdist[g_of, r_of] = ae.distance_km.to_numpy(dtype=np.float32)
        if e == "TA":
            scount = np.zeros(G, dtype=np.int8)
            scount[g_of] = ae.station_count.to_numpy().clip(0, 127)

        # F2: 가까운 순으로 값이 있는 첫 후보
        vals = np.where(cidx[:, :, None] >= 0, V[e][cidx.clip(0)], np.nan)                    # (G, 5, Ho)
        has = ~np.isnan(vals)
        first = np.argmax(has, axis=1)                                                        # (G, Ho)
        anyv = has.any(axis=1)
        gi = np.arange(G)[:, None]
        sel = np.where(anyv, vals[gi, first, np.arange(Ho)[None, :]], np.nan).astype(np.float32)
        sel_st = np.where(anyv, cidx[gi, first], -1)
        dist = np.where(anyv, cdist[gi, first], np.nan).astype(np.float32)
        if e == "WD":                                                                         # F4: 그 관측소의 풍속
            ws_same = np.where(sel_st >= 0, V["WS"][sel_st.clip(0), np.arange(Ho)[None, :]], np.nan)
            sel = np.where(ws_same < CALM, np.nan, sel)
        chk = np.where(sel_st >= 0, CHK[e][sel_st.clip(0), np.arange(Ho)[None, :]], False)

        # F3: 직전 값 최대 3시간
        age = np.where(np.isnan(sel), -1, 0).astype(np.int8)
        if e in CARRY:
            for h in range(1, Ho):
                miss = np.isnan(sel[:, h]) & (age[:, h - 1] >= 0) & (age[:, h - 1] < CARRY_H)
                if e == "WD":                                                                 # 무풍으로 비운 칸은 이어 쓰지 않음
                    miss &= ~(ws_same[:, h] < CALM)
                sel[miss, h] = sel[miss, h - 1]
                dist[miss, h] = dist[miss, h - 1]
                age[miss, h] = age[miss, h - 1] + 1
        V_sel = sel
        cur = V_sel[:, end_idx]
        track_dist[e] = dist[:, end_idx]
        track_age[e] = age[:, end_idx]
        chk_used += chk[:, end_idx].astype(np.int8)
        out[e] = cur
        if e == "HM":
            hm_series = V_sel
        if e == "TA":
            ta_series = V_sel
        if e == "WS":
            ws_series = V_sel
        if e == "RN_HR1":
            rn_series = V_sel

    # 누적 Feature (F5)
    feat = {
        "ta": out["TA"], "hm": out["HM"], "ws": out["WS"],
        "wd_sin": np.sin(np.deg2rad(out["WD"])).astype(np.float32),
        "wd_cos": np.cos(np.deg2rad(out["WD"])).astype(np.float32),
        "pa": out["PA"], "ps": out["PS"], "rn_1h": out["RN_HR1"], "rn_day": out["RN_DAY"],
        "ta_max_24h": window_stat(ta_series, end_idx, 24, "max"),
        "ta_min_24h": window_stat(ta_series, end_idx, 24, "min"),
        "hm_min_24h": window_stat(hm_series, end_idx, 24, "min"),
        "hm_mean_24h": window_stat(hm_series, end_idx, 24, "mean"),
        "ws_max_24h": window_stat(ws_series, end_idx, 24, "max"),
        "ws_mean_6h": window_stat(ws_series, end_idx, 6, "mean"),
        "rn_sum_24h": window_stat(rn_series, end_idx, 24, "sum"),
        "rn_sum_72h": window_stat(rn_series, end_idx, 72, "sum"),
    }
    # 건조 지속 시간: t-1시간부터 거슬러 1시간 강수 0.1 mm 미만이 이어진 시간 (최대 120). 빈 값은 건너뛰고 셈
    dry = np.zeros((G, Ho), dtype=np.int16)
    for h in range(Ho):
        r = rn_series[:, h]
        prevd = dry[:, h - 1] if h else np.zeros(G, dtype=np.int16)
        dry[:, h] = np.where(r >= 0.1, 0, np.where(np.isnan(r), prevd, np.minimum(prevd + 1, LB)))
    feat["dry_hours"] = dry[:, end_idx].astype(np.float32)
    if not os.path.exists(f"{S}/aws_silver_{prev}.parquet"):                                  # 첫 달: 120시간 앞 자료가 없음
        feat["dry_hours"][:, :LB] = np.nan
    # F7 실효습도
    he = np.zeros((G, H), dtype=np.float32)
    for k in range(5):
        he += (R_EFF ** k) * window_stat(hm_series, end_idx - 24 * k, 24, "mean")   # 하루라도 빠지면 He도 빈 값
    feat["eff_hm"] = (1 - R_EFF) * he
    for e in ELEM:
        feat[f"{e.lower()}_dist_km"] = track_dist[e]
        feat[f"{e.lower()}_age_h"] = track_age[e].astype(np.float32)
    feat["n_check_used"] = chk_used.astype(np.float32)

    # 각 행정동의 유효 기간 안 시각만 펼침
    vf = groups.valid_from.to_numpy()
    vt = groups.valid_to.to_numpy()
    rv = ref.to_numpy()
    inr = (rv[None, :] >= vf[:, None]) & (rv[None, :] < vt[:, None])
    gg, hh = np.nonzero(inr)
    df = pd.DataFrame({"admin_dong_code": groups.admin_dong_code.to_numpy()[gg],
                       "forecast_reference_time": ref[hh],
                       "station_count": scount[gg]})
    for k, v in feat.items():
        df[k] = v[gg, hh].astype(np.float32)
    for k in [c for c in df if c.endswith("_age_h")] + ["n_check_used", "dry_hours"]:
        df[k] = df[k].round().astype("Int16")
    df = df.sort_values(["admin_dong_code", "forecast_reference_time"]).reset_index(drop=True)
    assert not df.duplicated(["admin_dong_code", "forecast_reference_time"]).any(), "행정동 × 시각 중복"
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_parquet(f"{OUTDIR}/gold_feature_{m}.parquet", index=False)
    miss = df[["ta", "hm", "ws", "rn_1h", "hm_min_24h", "rn_sum_72h", "eff_hm"]].isna().mean().mul(100).round(2).to_dict()
    print(f"[{m}] {len(df):,}행 · 행정동 {df.admin_dong_code.nunique():,} · {time.time() - t0:.0f}초 · 빈 값(%) {miss}")
    return {"month": m, "rows": len(df), "dongs": df.admin_dong_code.nunique(),
            **{f"na_{k}": v for k, v in miss.items()},
            "ta_age_gt0": float((df.ta_age_h > 0).mean() * 100), "ta_rank_dist_med": float(df.ta_dist_km.median())}


if __name__ == "__main__":
    months = sys.argv[1:] or MONTHS
    assign = pd.read_parquet(f"{S}/stn_assign.parquet")
    log = [build_month(m, assign) for m in months]
    lg = pd.DataFrame(log)
    print("\n[요약]")
    print(lg.to_string(index=False))
    print(f"\n전체 {lg.rows.sum():,}행")
    os.makedirs("data/gold", exist_ok=True)
    if len(months) == len(MONTHS):
        lg.to_csv("data/gold/gold_feature_build_log.csv", index=False, encoding="utf-8-sig")