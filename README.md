# wildfire-ds — 산불 수치 데이터 프로젝트

행정동 × 1시간 단위 기상 Feature 데이터셋을 만들고, 산불 사건의 최종 피해면적을 예측할 수 있는지 Baseline으로 확인하는 과제. (DS기반 의사결정 지원, 2026-2)

## 먼저 읽을 것
- **온보딩 문서** (개요·개념·원천·절차·미결 이슈·재현 방법): https://claude.ai/artifact/GgtrGNeZE79sSdhmfMumLU
- Bronze 데이터 상세: [`data/bronze/README.md`](data/bronze/README.md)
- 데이터 파일(원본 zip, Parquet): 팀 드라이브 → (링크 추가)

## 현재 상태
- Bronze 완료 (2026-11-09): 기상 관측(AWS) 17,544시각 · 산불 사건 738건 · 관측소 744곳
- 다음: Silver (검증·법정동→행정동 매핑)

## 코드 구조
원천별로 `fetch_*.py`(수집) + `build_bronze_*_raw.py`(구조화) 한 쌍.

| 원천 | 수집 | 구조화 |
|---|---|---|
| 기상 관측 | `fetch_all.py` | `build_bronze_index.py`, `build_bronze_raw.py` |
| 산불 사건 | `fetch_fire.py` | `build_bronze_fire_raw.py` |
| 관측소 위치 | `fetch_stn.py` | `build_bronze_stn_raw.py` |
| 법정동·행정동 코드 | (행정안전부 게시판에서 직접 다운로드) | `build_bronze_adm_raw.py` |

## 실행 준비
1. `pip install requests python-dotenv pandas pyarrow openpyxl`
2. `.env.example`을 복사해 `.env`로 저장하고 인증키 입력 (기상청 API허브 키, 공공데이터포털 Decoding 키)
3. 프로젝트 루트에서 실행 (`.env`와 `data/` 경로가 실행 위치 기준)

데이터와 `.env`는 저장소에 올리지 않는다 (`.gitignore`).