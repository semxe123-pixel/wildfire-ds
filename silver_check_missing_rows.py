"""관측 행이 아예 없는 시각이 얼마나 있는지 확인 (저장 없음, 화면 출력만)

방법: 관측소 × 월마다 '그 달의 시간 수 - 실제 행 수' = 빠진 시각 수
      관측소가 처음 나온 시각 ~ 마지막 시각 사이만 셈 (신설·폐지로 생긴 앞뒤 빈 기간은 빼고 '중간에 빠진 시각'만)
입력  data/silver/aws_silver_YYYYMM.parquet (관측소, 관측 시각 열만 읽음)
"""
import glob
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

obs = pd.concat([pd.read_parquet(f, columns=["station_id", "observed_at"]) for f in sorted(glob.glob("data/silver/aws_silver_*.parquet"))],
                ignore_index=True)
g = obs.groupby("station_id").observed_at.agg(["min", "max", "size"])
g["span_h"] = ((g["max"] - g["min"]) / pd.Timedelta(hours=1)).astype(int) + 1     # 첫 시각 ~ 마지막 시각 사이 시간 수
g["missing_h"] = g.span_h - g["size"]                                              # 그 사이에 행이 없는 시각 수

print(f"전체 행 {len(obs):,} / 관측소 {len(g):,}")
print(f"시각마다 응답한 관측소 수: 최소 {obs.groupby('observed_at').size().min()}, "
      f"중앙값 {int(obs.groupby('observed_at').size().median())}, 최대 {obs.groupby('observed_at').size().max()}")
print(f"\n[중간에 빠진 시각] 있는 관측소 {int((g.missing_h > 0).sum())}곳 / 빠진 시각 합 {int(g.missing_h.sum()):,}시간 "
      f"(첫~마지막 시각 사이 전체의 {g.missing_h.sum() / g.span_h.sum() * 100:.2f}%)")
print("\n빠진 시각이 많은 관측소 10곳")
print(g.sort_values("missing_h", ascending=False).head(10)[["min", "max", "size", "missing_h"]].to_string())
gap = obs.sort_values(["station_id", "observed_at"])
d = gap.groupby("station_id").observed_at.diff().dt.total_seconds().div(3600)
print("\n빠진 구간의 길이(연속으로 빠진 시간) 분포:", (d[d > 1] - 1).describe(percentiles=[.5, .9, .99]).round(1).to_dict())