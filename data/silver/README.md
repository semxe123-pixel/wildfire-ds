# Silver 결과물 설명

Silver = Bronze의 글자 표를 계산할 수 있는 표로 바꾸고, 모든 데이터가 행정동으로 이어지게 만들고, 바꾼 내용을 남기는 단계.
규칙과 근거는 `docs/silver_plan.md`, 단계별 과정은 작업 기록 문서에 있다. 이 문서는 결과 파일마다 무엇이 들어 있는지를 적는다.

## 공통 기준 (2026-11-22)
- 시각: 모든 시각·날짜에 한국 시간 표시 (`Asia/Seoul`, 예: `2024-03-01 13:00+09:00`)
- 판정: 요소마다 `qc_요소` (0 정상 / 1 결측(기상청 -99) / 2 확인 필요(값 유지, 표시만) / 3 제거(그 칸만 빈 값)) + `qc_요소_flags` (걸린 규칙 번호를 쉼표로 이음). 행은 지우지 않음. 원래 값은 Bronze에 남음
- 이용 가능시각: `available_at` = `observed_at` + 1시간 (예측 시점에 아직 받지 못한 값을 쓰지 않기 위함)
- 빈 값 채우기(대체)는 Silver에서 하지 않음 → Gold
- 열 이름: 설명서(Gold 6절)의 표준명을 Silver부터 사용. 관측시각 `observed_at`, 이용 가능시각 `available_at`, 관측소 `station_id`(문자열), 원본 경로 `bronze_file`, 행정동 `admin_dong_code`·`admin_dong_name`. 표준명이 없는 법정동 쪽은 같은 방식으로 `legal_dong_code`·`legal_dong_name`
- 원천 추적: 모든 표에 `bronze_file`
- 옛 결과물(2026-11-16)은 `data/_archive/silver_20261116/`에 비교용으로 보관. 새 결과를 검증한 뒤 지움

# 변환표

## 파일
- `adm_map.parquet` 58,966행. 코드 `build_silver_adm_map.py`
- `adm_changes_2024_2025.csv` 3,908행 (UTF-8 BOM). 같은 코드

## 원천
- `data/bronze/adm_raw.parquet` 중 KIKmix, 20261026 말소판 (`jscode20261026_malso`) 단일 (규칙 A4)
- 비교용: 20260101 일반판 KIKmix (검증 ①에만 사용)

## adm_map 열 (설명서 표준명 기준)
- admin_dong_code, legal_dong_code (문자열 10자리), sido_name, sigungu_name, admin_dong_name(행정동 이름), legal_dong_name(법정동·리 이름)
- valid_from, valid_to: 관계의 유효기간. 한국 시간 00:00. valid_to가 비어 있으면 지금도 유효
- admin_level (sido / sigungu / emd), legal_level (sido / sigungu / emd / ri): 코드 자릿수로 구분
- bronze_file: Bronze 원본 경로 (`adm/폴더/파일`)
- 날짜 d에 유효한 관계 = valid_from ≤ d AND (valid_to 없음 OR valid_to > d) (규칙 A1). 매핑에는 admin_level == emd만 사용 (규칙 A2)

## adm_changes_2024_2025 열
- date: 관계가 생기거나 끝난 날, change: start / end
- code_only: 같은 날 이름이 같은 관계가 반대쪽에 있으면 True (코드만 바뀜). False = 이름도 바뀜 (이름 변경, 분동, 합동 등)
- sido_name, sigungu_name, admin_dong_code, admin_dong_name, legal_dong_code, legal_dong_name
- 범위: 2024-01-02 ~ 2025-12-31에 생기거나 끝난 읍면동 수준 관계. 생성일 = 말소일인 행 제외

## 검증 (2026-11-22)
- 검증 ① 2026-01-01 유효 관계 vs 20260101 일반판: 말소일자 미포함 해석에서 말소판에만 0 / 일반판에만 1 (화성 반월동↔동탄1동, 규칙 A4에 기록). 옛 결과와 같음
- 검증 ② 2025-03-22 법정동 20,308개 중 행정동 2개 이상 713개 (규칙 A3). 옛 결과와 같음
- 기간 중 변경 3,908행. 그중 3,802행은 2024-01-18 전북 코드 변경(45→52, 코드만 바뀜)

## 경계 기준일 사이 행정동 변화 (이름 기준)
- 2024-01-01 → 2024-11-30: 3,620 → 3,621. 없어짐 2 (성주군 금수면, 예천군 호명면), 생김 3 (의정부시 고산동, 성주군 금수강산면, 예천군 호명읍)
- 2024-11-30 → 2025-12-31: 3,621 → 3,625. 없어짐 9, 생김 13 (안양 박달동·호현동, 용인 유림1·2동, 인천 아라1·2동, 전주 금암동·송천3동, 동대문 신설동·용두동, 부산 신호동 등)
- → 행정동 경계 파일 2개(2024-11, 2025-12)만으로는 기간 전체를 그대로 덮을 수 없음. 이름만 바뀐 곳(금수강산면, 호명읍)은 땅이 같으므로 문제없음. 분동·합동된 지역은 관측소 매핑에서 따로 확인

## 한계
- 변환표가 혼자 답을 내는 범위는 "코드가 있고, 그 날짜에 행정동이 하나뿐인 경우"까지 (2025-03-22 기준 법정동 20,308개 중 19,595개)
- 법정동 하나가 행정동 여러 개에 걸침 (713개, 규칙 A3) → 관측소는 좌표 + 경계, 산불은 좌표 또는 따로 정할 규칙
- 그 날짜에 유효한 줄이 없는 코드 (관측소 LAW_ID 중 2026 새 코드, 2022 폐지된 오포읍 코드 등), 시군구까지만 적힌 코드 (관측소 3곳) → 좌표로 판정
- 산불 주소는 코드가 아니라 이름(약칭)이라 이름 → 코드 변환이 따로 필요
- 변환표에는 경계선이 없음. 코드가 같아도 경계가 바뀔 수 있음 (설명서) → 경계 파일을 버전·유효기간과 함께 사용
- 말소판과 공식 일반판을 대조한 날짜는 2026-01-01 하나뿐. 2024~2025 날짜를 대조할 자료는 없음
- code_only 판정은 이름 비교라서, 전북 외의 이름 변경이 기간 중에 있으면 "이름도 바뀜"으로 분류될 수 있음

# 기상 (AWS 매시)

## 파일 (2026-11-23)
- `aws_silver_YYYYMM.parquet` 24개, 12,549,269행. 코드 `build_silver_aws.py`
- `aws_availability.parquet` / `.csv` 138,744행 (관측소 × 월 × 요소)
- `aws_qc_summary.csv` 규칙별 건수 (2024·2025년 따로)
- `aws_units.csv` 요소별 이름·단위·설명

## 원천
- `data/bronze/aws_raw/aws_raw_YYYYMM.parquet` 24개월 (기상청 API허브 `awsh.php` 매시 정시 자료)
- 관측소 좌표·높이: `data/bronze/stn_raw.parquet`. 그 달 1일 지점정보, 없으면 그 뒤 첫 지점정보
- 원자료에 기상청 품질 필드(QCM)와 관측시각 차이 필드는 없음 (원본 텍스트에서도 확인)

## aws_silver 열
- observed_at (한국 시간), available_at (= observed_at + 1시간), station_id (문자열)
- TA 기온 ℃, WD 풍향 deg, WS 풍속 m/s, RN_DAY 일강수량 mm, RN_HR1 1시간 강수량 mm, HM 상대습도 %, PA 현지기압 hPa, PS 해면기압 hPa (float32. 판정 3 제거는 빈 값)
- qc_TA … qc_PS: 판정 0~3 (int8)
- qc_TA_flags … qc_PS_flags: 걸린 규칙 번호 (예: `3-2,3-3`). 걸린 규칙이 없으면 빈 값. 규칙 번호와 내용은 `docs/silver_plan.md`
- bronze_file: Bronze 원본 경로 (`aws/YYYYMMDDHHMI.txt`)

## 판정 결과 (행)
| 요소 | 정상 | 결측 | 확인 필요 | 제거 |
| --- | --- | --- | --- | --- |
| TA | 12,527,554 | 9,398 | 3,043 | 9,274 |
| WD | 12,292,481 | 256,788 | 0 | 0 |
| WS | 12,368,923 | 180,328 | 18 | 0 |
| RN_DAY | 12,192,322 | 356,609 | 338 | 0 |
| RN_HR1 | 12,173,556 | 375,149 | 8 | 556 |
| HM | 12,190,544 | 356,748 | 1,959 | 18 |
| PA | 9,981,824 | 2,559,806 | 4,285 | 3,354 |
| PS | 9,969,030 | 2,575,855 | 4,317 | 67 |

- 산불에 중요한 극한값: 습도 15% 미만 20,796행 중 제거 18 (모두 습도 0% 기록 오류), 풍속 10 m/s 이상 93,680행 중 제거 0, 기온 35℃ 이상 21,085행 중 제거 1 (120.4℃)

## aws_availability 열
- station_id, month (YYYYMM), element
- n_rows (그 달 행 수), n_normal, n_missing, n_check, n_removed
- hours_in_month (그 달 전체 시간), value_share = (n_normal + n_check) / hours_in_month
- 값이 하나도 없는 관측소×월: TA 2, WD 91, WS 46, RN_DAY 264, RN_HR1 269, HM 312, PA 3,515, PS 3,520 → 요소마다 관측하는 관측소가 다름. 대표 관측소는 요소별로 고름 (관측소 매핑)

## 한계
- 실시간 원칙 때문에 고착 구간의 앞부분(예: 24시간 기준이면 1~23시간째)은 표시되지 않음
- 335~340번: 2024-12 신설, 관측값 2024-12부터. 처음 약 한 달(2024-12~2025-01)은 해면기압이 현지기압과 맞지 않음 (규칙 4-4 확인 필요)
- 358번의 긴 무풍 구간은 이웃 바람이 약해(약 1 m/s) 판정하지 않음
- 계단형 오류(값이 한 번 바뀐 뒤 유지), 기후범위(관측소·월별 정상 범위)는 검사하지 않음
- 고산 관측소의 아주 낮은 습도는 진짜일 수 있어 높이가 비슷한 이웃과만 비교함 (규칙 3-6)
- 풍향은 풍속 0.2 m/s 미만이면 의미 없음 (WMO). Silver에서는 그대로 두고 Gold에서 제외

# 관측소 매핑 (관측소 위치, 소재지, 행정동 대표값 배정)

## 파일 (2026-11-23)
- `stn_month.parquet` / `.csv` 17,596행 (관측소 × 월). 코드 `build_silver_stn_month.py`
- `stn_adm.parquet` / `.csv` 17,641행 (관측소 × 날짜 범위). 코드 `build_silver_stn_adm.py`
- `adm_point.parquet` / `.csv` 85,675행 (행정동 × 날짜 범위). 코드 `build_silver_stn_assign.py`
- `stn_assign.parquet` 3,421,620행 (행정동 × 날짜 범위 × 요소 × 순위). 같은 코드. 행이 많아 csv 없음
- 옛 `stn_to_adm.parquet`(2026-11-16, 2시점·관측소당 위치 하나)는 쓰지 않음
- 규칙 번호(P, B, D, R)와 근거는 `docs/silver_plan.md`

## 원천
- 관측소: `data/bronze/stn_raw.parquet` (지점정보 25시점: 2024-01~2025-12 매달 1일, 2025-12-31 23시)
- 행정동 경계: `data/bronze/boundary/202411/`, `202512/` (행정안전부 "구역의 도형" 시도 17개 zip × 2시점, 그중 `TL_SCCO_GEMD` 행정동 층). 좌표계 정보가 없어 EPSG:5179로 확인, 글자 cp949
- 코드 유효기간: `adm_map.parquet` / 후보 자격: `aws_availability.parquet`

## 날짜 범위
- `valid_from` 이상, `valid_to` 미만 (한국 시간). 시각 t의 행을 찾을 때 `valid_from <= t < valid_to`
- 보통 한 달이 한 행. 그 달 안에서 행정동 코드가 바뀐 경우만 그 날짜로 나뉨 (예: 전북 2024-01-18)

## stn_month 열
- station_id, month (YYYYMM), lon, lat (WGS84), height_m, station_name
- position_snapshot (쓴 지점정보 시점), position_source (first_of_month 그 달 1일 / next 달 중간 신설이라 다음 지점정보)
- moved_in_month (다음 지점정보와 0.1 km 이상 차이), move_km_next
- has_aws_data (그 달 기상 자료가 있는지), law_id_raw, law_addr_raw (지점정보 원문, 2026년 코드·이름이 섞여 있어 검증용), bronze_file

## stn_adm 열 (소재지: 이 관측소는 그날 어느 행정동에 있나)
- station_id, valid_from, valid_to, month
- station_admin_dong_code, admin_dong_name, sigungu_name, sido_name: 그날 유효한 소재지 행정동
- spatial_mapping_method: within (구역 안) / nearest (구역 밖, 가장 가까운 구역) / address (구역 밖, 주소로 정함)
- code_mapping: boundary (경계 코드 그대로) / legal_bridge / name_bridge / other_boundary / address
- boundary_admin_dong_code: 경계 파일에서 나온 원래 코드 (코드를 바꿨으면 비교용)
- is_estimated: within이고 boundary가 아니면 True
- mapping_confidence: 관측소 둘레 반경 100 m 원 중 그 구역 안에 드는 넓이 비율 (0~1). distance_to_boundary_m: 구역 밖이면 구역까지 거리
- admin_boundary_version, admin_boundary_valid_from, admin_boundary_valid_to: 쓴 경계 파일과 그 사용 기간
- lon, lat, moved_in_month, has_aws_data, position_snapshot, stn_bronze_file, boundary_file

결과 (기상 자료가 있는 17,388행): within 17,080 · nearest 284 · address 24 / boundary 17,313 · name_bridge 45 · address 24 · other_boundary 6 / is_estimated 356 / 신뢰도 1.0인 행 82.7%

## adm_point 열 (행정동 대표점)
- admin_dong_code, valid_from, valid_to, month, admin_dong_name, sigungu_name, sido_name
- admin_centroid_lon, admin_centroid_lat (WGS84), centroid_method (centroid 무게중심 / inside_point 무게중심이 구역 밖이라 구역 안의 점), area_km2
- code_mapping: 구역 코드를 그날 코드로 맞춘 방법 (boundary, legal_bridge, name_bridge, overlap_bridge, overlap_split, sigungu_pair. 여러 구역을 합쳤으면 쉼표로 이음)
- n_polygons (합친 구역 수), polygon_codes (원래 구역 코드), polygon_source (same 그 기간 경계 파일 / other_boundary 다른 경계 파일), trimmed (다른 경계 파일 구역과 겹친 만큼 깎였는지)
- admin_boundary_version, admin_boundary_valid_from, admin_boundary_valid_to, boundary_file

## stn_assign 열 (행정동 값을 채울 후보 관측소)
- admin_dong_code, valid_from, valid_to, month, element (TA, WD, WS, RN_DAY, RN_HR1, HM, PA, PS)
- rank (1~5, 가까운 순), station_id, distance_km (대표점에서 관측소까지)
- station_height_m, station_outside (구역 밖 섬·바다 관측소)
- station_count: 반경 30 km 안 후보 관측소 수 (같은 행정동·범위·요소에서 모든 순위가 같은 값)
- admin_boundary_version
- 후보 자격: 그 달 그 요소 값이 하나라도 있는 관측소, 대표점에서 30 km 안

## Gold에서 쓰는 법
- 행정동 c, 시각 t, 요소 e → `stn_assign`에서 c·e이고 `valid_from <= t < valid_to`인 행을 rank 순으로 봄
- `aws_silver`에서 그 관측소의 시각 t 값이 있는(판정 1 결측·3 제거가 아닌) 첫 후보의 값을 씀
- `station_distance_km` = 그 시각 실제로 쓴 후보의 distance_km, `station_count` = station_count
- 관측소 → 소재지 행정동이 필요하면 `stn_adm`에서 같은 방식으로 날짜 범위를 찾음

## 한계
- 행정동 높이 자료가 없어 고도 차이(100 m당 약 0.65℃)를 보정하지 않음
- 경계 파일이 두 시점뿐이라 기간 중 바뀐 행정동은 규칙으로 메움 (R1~R4, `docs/silver_plan.md`). 해당 행정동은 code_mapping·polygon_source·trimmed로 확인 가능
- 소재지 검증(시군구 이름이 주소에 있나) 불일치 15곳: 세종(시군구 없음, 검증 코드 문제), 인천 2026-07 새 구 이름, 산지 시군 경계 875·694(좌표가 구역 안이라 좌표를 따름), 682(주소가 군사분계선 북쪽인 철원군 임남면)
- 군사분계선 북쪽 행정동(임남면 등)도 경계 파일에 있어 대표점·후보가 있음 → Gold 범위에서 결정