# Bronze — 기상 관측 (AWS 시간통계) 원본 보존

## 1. 원천
- 제공: 기상청 API허브 (https://apihub.kma.go.kr)
- API: AWS 시간통계 자료 조회 `api/typ01/url/awsh.php`
- 호출 방식: REST GET, 인자 `tm`(시각), `help=0`, `authKey`. `var` 생략 → 정시자료 8개 요소
- 1회 호출 = 1시각 × 전국 AWS 관측소 전체 (`stn` 미지정)
- 응답 형식: 공백 구분 텍스트. `#` 줄은 주석, 본문은 `#START7777` ~ `#7777END`
- 여기서 AWS = Automatic Weather Station(자동기상관측장비)

## 2. 수집 범위
- 기간: 2024-01-01 00:00 ~ 2025-12-31 23:00 (KST), 1시간 간격 → 17,544시각
- 수집일: 2026-11-08 14:08 ~ 19:44 (약 5.5시간)
- 수집 코드: `fetch_all.py` (시각 목록 생성 → 파일 있으면 건너뜀 → 최대 3회 재시도 → 0.3초 간격)
- 인증키는 `.env`의 `KMA_AUTH_KEY`. 저장소·공유 파일에 포함하지 않음

## 3. 폴더 구조
```
data/bronze/
├── aws/                  원본 응답 txt 17,544개. 파일명 = tm (yyyymmddhhmi). 수정 금지
├── aws_index.parquet     수집 인덱스 (파일 1개 = 1행)
├── aws_index.csv         위와 동일 내용, 열람용
└── aws_raw/              원천 구조화 표. 월별 Parquet 24개 (aws_raw_YYYYMM.parquet)
```

## 4. 수집 인덱스 (aws_index) — 17,544행
| 열 | 내용 |
|---|---|
| file | 원본 파일명 |
| tm | 요청 시각 |
| source | apihub.kma.go.kr/api/typ01/url/awsh.php |
| params | 요청 인자 (`tm=...&help=0`, authKey 제외) |
| collected_at | 수집 시각. **파일 mtime 기준** (수집 당시 별도 기록하지 않아 사후 생성) |
| size_bytes | 파일 크기 |
| sha256 | 파일 checksum |
| n_rows | 데이터 행 수 (`#` 아닌 줄) |
| has_end | `#7777END` 존재 여부 |

생성 코드: `build_bronze_index.py`

## 5. 원천 구조화 표 (aws_raw) — 12,549,269행
- 열 11개: 원천 헤더 10개 + `src_file`
  `YYMMDDHHMI, STN, TA, WD, WS, RN_DAY, RN_HR1, HM, PA, PS, src_file`
- 모든 값은 **문자열 그대로** (자료형 변환·결측 처리 없음). 열 이름은 파일 헤더 그대로
- `src_file`: 해당 행이 나온 원본 파일명 (인덱스 `file`과 대응)
- 토큰 수가 10이 아닌 줄은 수정하지 않고 `aws_raw_badlines.csv`에 기록하도록 함 → 해당 없음(0건)
- 생성 코드: `build_bronze_raw.py`

원천 열 의미 (API 설명서 기준)
| 열 | 의미 | 단위 |
|---|---|---|
| YYMMDDHHMI | 관측 시각 (KST) | yyyymmddhhmi |
| STN | 관측소 번호 | — |
| TA | 정시 기온 | ℃ |
| WD | 정시 10분 평균 풍향 | degree |
| WS | 정시 10분 평균 풍속 | m/s |
| RN_DAY | 해당 시각까지 일강수량 | mm |
| RN_HR1 | 1시간 강수량 | mm |
| HM | 정시 습도 | % |
| PA | 정시 현지기압 | hPa |
| PS | 정시 해면기압 | hPa |

## 6. 완료 기준 검증 (2026-11-09)
| 기준 | 결과 |
|---|---|
| 파일 수 | 17,544 (= 기간 내 시각 수) |
| tm 중복 | 0 |
| 불완전 응답 (`#7777END` 없음) | 0 |
| 인덱스 n_rows 합계 | 12,549,269 |
| 구조화 표 행 수 | 12,549,269 |
| 이상 행 (토큰 ≠ 10) | 0 |
| 건수 일치 | 일치 |

## 7. 확인된 원천 특성 (Silver에서 처리할 항목)
- 결측 표기: `-99.0` / `-99` (주로 PA·PS)
- 시각당 관측소 수 645 ~ 727로 변동 (신설·폐지·장애). 관측소별 결측 패턴 확인 필요
- 일부 관측소에서 TA `0.0`이면서 PA·PS가 -99인 행 존재 → `0.0`이 실질 결측일 가능성 확인 필요
- 행 순서는 STN 번호순이 아님. 순서에 의존하지 말 것
- 위치 정보 없음. STN → 위경도 → 행정동 변환을 위한 관측소 지점 정보 원천 별도 필요

## 8. 재수집·재생성 방법
```
python fetch_all.py            # 없는 시각만 받음 (기존 파일 건너뜀)
python build_bronze_index.py   # 인덱스 재생성
python build_bronze_raw.py     # 구조화 표 재생성 + 건수 검증
```

## 9. 공유
- 코드: GitHub (`data/`, `.env`, `*.zip`은 `.gitignore`로 제외)
- 원본 txt: zip으로 압축해 팀 드라이브 1벌 보관 (`bronze_aws_2024_2025.zip`)
- 팀원 작업용: `aws_raw/` Parquet 및 `aws_index.parquet`

# Bronze — 산불 사건 (산림청 산불발생통계)

## 원천
- 공공데이터포털 `산림청_산불발생통계(대국민포털)` (data.go.kr/data/3070842)
- 요청주소: https://apis.data.go.kr/1400000/forestStusService/getfirestatsservice
- REST GET, 응답 XML. 인자: ServiceKey, numOfRows(99), pageNo, searchStDt, searchEdDt(발생일 기준)
- 인증키는 `.env`의 `DATA_GO_KR_KEY` (Decoding 키)

## 수집
- 범위: 발생일 2024-01-01 ~ 2025-12-31, 월 단위 조회 × 페이지
- 수집일: 2026-11-09. 코드 `fetch_fire.py` (수집과 동시에 인덱스 기록)
- 파일: `fire/fire_YYYYMM_pNN.xml` 25개 (응답 1페이지 = 1파일, 원본 그대로)
- 인덱스: `fire_index.parquet` (file, source, params, collected_at, size, sha256, result_code, total_count, n_items)
- 구조화 표: `fire_raw.parquet` 738행 × 17열 (원천 태그 16 + src_file, 전부 문자열). 코드 `build_bronze_fire_raw.py`

## 검증
- 월별 totalCount 합 738 == item 합 738 == 구조화 표 738행
- 2024년 279건, 2025년 459건 (2025년은 산림청 발표 연간 건수와 일치)

## 원천 특성 (Silver 처리 항목)
- 사건 고유 ID 없음 → 발생일시+주소 조합으로 키 생성, 중복 확인 필요
- `locmenu`(읍면) 태그가 동 지역 107건에서 부재 → 빈 문자열로 보존
- `locsi` 약칭(경남, 전북 등) → 정식 명칭 변환 필요. 제주 0건
- `damagearea` 단위 미표기 (ha로 추정, 검증 예정). 피해액·인명 필드 없음 → Label은 피해면적만
- 3~4월 집중(2년 367건), 7~10월 희소

# Bronze — AWS 관측소 지점정보

## 원천
- 기상청 API허브 `typ01/url/stn_inf.php?inf=AWS` (지상관측 지점정보 조회)
- 인자: inf=AWS, stn=(전체), tm=시점, help=1, authKey
- 응답: 고정폭 텍스트. `#` 설명 줄 + 헤더 2줄 + 데이터

## 수집
- 시점 2개: tm=202401010000, tm=202512312300 (수집 기간 양 끝). 수집일 2026-11-09
- 파일: `stn/stn_aws_{tm}.txt` 2개. 코드 `fetch_stn.py`
- 인덱스: `stn_index.parquet`. 구조화 표: `stn_raw.parquet` 1,466행 × 15열 (원천 14열 + src_file, 전부 문자열). 코드 `build_bronze_stn_raw.py` (정규표현식으로 고정폭 분리)
- 열: STN_ID, LON, LAT, STN_SP, HT, HT_WD, LAU_ID, STN_AD, STN_KO, STN_EN, FCT_ID, LAW_ID, BASIN, LAW_ADDR
  (허브 설명서 표의 HT_PA/HT_TA/HT_RN/STN_CD는 실제 응답에 없음. 파일 헤더를 따름)

## 검증
- 2024초 727 / 2025말 739 / 합집합 744. 폐지 5, 신설 17. 파싱 실패 0. LAW_ID 10자리 전부 유효

## 원천 특성 (Silver 처리 항목)
- BASIN이 `----`인 관측소 존재(공항). STN_EN은 대부분 `----`
- BASIN과 LAW_ADDR 사이 공백 없음, 일부 LAW_ADDR 앞에 `(산지)` 접두
- **LAW_ADDR·LAW_ID는 tm과 무관하게 현재(2026) 행정구역 체계** (예: 전남광주통합특별시, 코드 12xx).
  "발생 당시 행정동" 매핑 시 코드 체계 연도 불일치 가능 → 법정동→행정동 변환표의 기준 연도와 맞출 것
- 시점 간 관측소 증감 있음 → AWS Bronze의 STN 집합이 합집합 744에 포함되는지 Silver에서 확인

# Bronze — 법정동·행정동 코드 (행정안전부)

## 원천
- 행정안전부 누리집 > 업무안내 > 자치혁신실 > 주민등록,인감 게시판
  "행정기관(행정동) 및 관할구역(법정동) 변경내역" 게시물의 첨부 zip
- 게시물 2개: 2026.1.1. 시행 (2025년 말 스냅샷), 2026.10.30. 기준 알림 (최신, 파일명 20261026)
- zip 안: KIKcd_H(행정동코드), KIKcd_B(법정동코드), KIKmix(행정동–법정동 관할 관계). 텍스트 + xlsx 동봉.
  "(말소코드포함)" 버전은 폐지된 코드·관계를 말소일자와 함께 포함

## 수집 (2026-11-16, 수동 다운로드)
- `adm/jscode20260101.zip`, `adm/jscode20260101(말소코드포함).zip`, `adm/jscode20261026(말소코드포함).zip`, Layout.hwpx
- 인덱스: `adm_index.parquet` (file, source 게시물, collected_at=다운로드 mtime, size, sha256)
- 압축 해제본: `adm/jscode20260101/`, `adm/jscode20260101_malso/`, `adm/jscode20261026_malso/` (zip이 원본, 해제본은 입력용. 말소 zip은 PowerShell 압축 해제 시 한글 파일명이 깨지나 내용 무관)
- 구조화 표: `adm_raw.parquet` 282,442행. xlsx 9개를 문자열 그대로 + table(KIKmix/KIKcd_H/KIKcd_B), snapshot, src_file. 코드 `build_bronze_adm_raw.py`

## 열 (KIKmix)
행정동코드(10), 시도명, 시군구명, 읍면동명(행정동명), 법정동코드(10), 동리명, 생성일자(yyyymmdd), 말소일자(yyyymmdd 또는 빈값)

## 검증·특성
- 20260101 일반판 KIKmix 21,816행 말소 0 / 말소판 55,220행 말소 33,404 / 20261026 말소판 58,966행 말소 37,125
- 2024-01-01~2025-12-31 사이 말소된 관계 2,015행 (두 말소판 동일) → 날짜별 유효 매핑 필터 필수
- 2026-07-01 이후 생성 3,505행, 2026-01 이후 말소 3,721행 = 7월 행정구역 개편(광주·전남·인천·안양)
- 20261026 말소판이 20260101 말소판의 상위집합 → Silver 변환표는 20261026 말소판 기준, 20260101은 교차 검증용