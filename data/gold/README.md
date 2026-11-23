# Gold (2026-11-23)

행정동 × 1시간 날씨 Feature 표와, 산불 사건마다 발생 시점 Feature + 피해 Label을 붙인 학습 표.
변수 정의서: https://claude.ai/artifact/53DtWEoGJT3nqsrJcqigqp (Word·PDF로 내려받기 가능)

## 파일
| 파일 | 행 | 만든 코드 |
|---|---|---|
| `feature/gold_feature_YYYYMM.parquet` (24개) | 62,445,288 (행정동 × 기준 시각) | `build_gold_feature.py` |
| `gold_feature_build_log.csv` | 24 (월별 행 수·빈 값) | `build_gold_feature.py` |
| `gold_train.parquet` / `.csv` | 734 (산불 사건) | `build_gold_train.py` |
| `gold_report.md` | Gold 검증 보고 (R1~R7) | `build_gold_report.py` |
| `gold_feature_stats.csv` | 열별 빈 값·최소·분위수·최대 | `build_gold_report.py` |

`feature/`는 수 GB라 커밋하지 않음. 코드로 다시 만들 수 있음 (한 달 약 15초).

## 규칙
- F1 기준 시각 t의 값 = 관측 시각 t-1시간의 관측 (`available_at` = 관측 + 1시간 ≤ t). 누출 방지
- F2 후보 관측소(`stn_assign`, 30 km 안 최대 5곳)를 가까운 순으로 보고 값이 있는 첫 관측소 값. 판정 0·2 사용, 1·3 건너뜀
- F3 5곳 모두 없으면 직전 값 최대 3시간 (기온·습도·풍속·풍향·기압). 강수는 이어 쓰지 않음. 앞뒤 보간 금지
- F4 풍속 0.2 m/s 미만이면 풍향 빈 값. 풍향은 sin·cos
- F5 누적 값은 t-1시간에서 끝나는 지난 W시간, 창 안 값 75% 이상일 때만
- F6 과거 창(최대 120시간)도 그 기준 시각의 후보 관측소로 계산
- F7 실효습도 He = 0.3 × Σ 0.7^k × Hk (k = 0~4, Hk = k일 전 24시간 평균 습도)
- F8 고도 보정 없음, 쓴 관측소 거리(`{요소}_dist_km`)를 남김
- T1 학습 표 기준 시각 = 발생 시각을 정시로 내림 · T2 행정동 못 정한 4건 제외 · T3 시간 Feature, 바람 벡터(무풍 = 0)
- T4 Label: damage_area_ha, log_damage_area, deaths, injuries, property_damage_kkrw(천원), has_casualty, duration_h
- T5 발생 뒤에야 아는 열은 `post_` 접두어 (Feature 금지) · T6 split: 2024 train / 2025 test

## 검증 결과 (`gold_report.md`)
- R1 월마다 행 수 = 후보 관측소 표의 (행정동 × 유효 시각) 수: 62,445,288 = 62,445,288
- R2 (행정동, 기준 시각) 중복 0 · R4 값 범위 벗어난 값 0
- R3 현재 값 빈 값 0.006~0.008%. 풍향 8.7%는 무풍. 누적 값 빈 값 0.1~0.7%는 2024-01 초반(앞 자료 없음)
- R5 직전 값 이어 쓴 비율 월 0.03% 이하, 확인 필요 값 사용 월 0.01~0.12%
- R6 무작위 240줄의 기온을 Silver에서 규칙대로 다시 계산: 240/240 일치
- R7 학습 표 734행 (train 279 / test 455), 기준 시각 ≤ 발생 시각. 누적 값 빈 값은 2024-01 초 사건 1~2건

## 한계
- 가장 가까운 관측소 한 곳의 값 (거리 중앙값 3.5 km), 고도 보정 없음
- 풍속은 정시 값이라 돌풍 미반영
- 실효습도는 24시간 단위 계산이라 기상청 발표값과 조금 다를 수 있음
- `dry_hours`는 120시간에서 멈춤 (학습 표 중앙값 120) → Baseline에서 창 확대 검토
- Baseline 모델은 2026-11-23 제출에서 제외, 다음 단계