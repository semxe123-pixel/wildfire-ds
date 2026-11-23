import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone, timedelta
from urllib.parse import parse_qsl
import pandas as pd

BRONZE = "data/bronze"
OUT = f"{BRONZE}/manifest"

# 원천별 이용조건. 확인한 날짜와 함께 적는다. 확인 전이면 "미확인".
LICENSE = {
    "kma": ("기상청 저작권정책: 기상청이 저작재산권 전부를 보유한 저작물은 자유이용, "
            "공공누리 제1유형(출처표시). API허브 개별 약관은 따로 확인하지 않음 (2026-11-22 확인)",
            "https://www.kma.go.kr/kma/guide/copyright.jsp"),
    "forest": ("이용허락범위 제한 없음 (공공데이터포털 표시, 2026-11-22 확인)",
               "https://www.data.go.kr/data/3070842/openapi.do"),
    "mois": ("미확인", ""),
}

COLUMNS = [
    "source_dataset", "provider", "endpoint", "request_method", "request_params_redacted",
    "source_schema_version", "requested_period_start", "requested_period_end",
    "collected_at", "collected_at_basis", "bronze_file", "sha256", "size_bytes",
    "record_count", "collection_status", "error_code",
    "source_temporal_resolution", "source_spatial_resolution", "license", "license_url",
    "index_file",
]


def params_json(qs):
    """'tm=202401010000&help=0' → '{"tm": "202401010000", "help": "0"}'. 인증키는 원래 들어 있지 않다."""
    d = dict(parse_qsl(qs, keep_blank_values=True))
    assert not any(k.lower() in ("authkey", "servicekey") for k in d), "인증키가 params에 있음"
    return json.dumps(d, ensure_ascii=False)


def kst(tm):
    """'202401010000' → '2024-01-01T00:00+09:00' (기상청 tm은 KST)"""
    return f"{tm[:4]}-{tm[4:6]}-{tm[6:8]}T{tm[8:10]}:{tm[10:12]}+09:00"


def ymd(d):
    """'20240101' → '2024-01-01'"""
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def aws():
    idx = pd.read_parquet(f"{BRONZE}/aws_index.parquet")
    lic, url = LICENSE["kma"]
    return pd.DataFrame({
        "source_dataset": "AWS 시간자료 정시값 (awsh.php, var 생략)",
        "provider": "기상청 API허브",
        "endpoint": "https://" + idx["source"],
        "request_method": "GET",
        "request_params_redacted": idx["params"].map(params_json),
        "source_schema_version": "공백 구분 10열: YYMMDDHHMI, STN + 정시 8요소 (2026-11-08 수집분)",
        "requested_period_start": idx["tm"].astype(str).map(kst),
        "requested_period_end": idx["tm"].astype(str).map(kst),
        "collected_at": idx["collected_at"].astype(str),
        "collected_at_basis": "파일 저장 시각 (수집 후 인덱스 생성)",
        "bronze_file": "aws/" + idx["file"],
        "sha256": idx["sha256"],
        "size_bytes": idx["size_bytes"],
        "record_count": idx["n_rows"],
        "collection_status": idx["has_end"].astype(str).eq("True").map({True: "success", False: "incomplete"}),
        "error_code": "",
        "source_temporal_resolution": "1시간 (정시 관측값)",
        "source_spatial_resolution": "관측소 지점 (STN)",
        "license": lic, "license_url": url,
        "index_file": "aws_index.parquet",
    })


def fire():
    idx = pd.read_parquet(f"{BRONZE}/fire_index.parquet")
    p = idx["params"].map(lambda s: dict(parse_qsl(s)))
    lic, url = LICENSE["forest"]
    code = idx["result_code"].astype(str).str.zfill(2)   # 숫자로 저장돼 있어도 "00"으로 맞춤
    return pd.DataFrame({
        "source_dataset": "산림청_산불발생통계(대국민포털) (공공데이터포털 3070842)",
        "provider": "산림청 (공공데이터포털)",
        "endpoint": idx["source"],
        "request_method": "GET",
        "request_params_redacted": idx["params"].map(params_json),
        "source_schema_version": "XML item 태그 16개 (2026-11-09 수집분)",
        "requested_period_start": p.map(lambda d: ymd(d["searchStDt"])),
        "requested_period_end": p.map(lambda d: ymd(d["searchEdDt"])),
        "collected_at": idx["collected_at"].astype(str),
        "collected_at_basis": "요청 시 기록",
        "bronze_file": "fire/" + idx["file"],
        "sha256": idx["sha256"],
        "size_bytes": idx["size_bytes"],
        "record_count": idx["n_items"],
        "collection_status": code.map(lambda c: "success" if c == "00" else "error"),
        "error_code": code.where(code != "00", ""),
        "source_temporal_resolution": "사건 (발생·진화 일시, 초 단위)",
        "source_spatial_resolution": "주소 글자 (시도 약칭, 시군구, 읍면, 동리, 지번)",
        "license": lic, "license_url": url,
        "index_file": "fire_index.parquet",
    })


def stn():
    idx = pd.read_parquet(f"{BRONZE}/stn_index.parquet")
    tm = idx["params"].map(lambda s: dict(parse_qsl(s, keep_blank_values=True))["tm"])
    lic, url = LICENSE["kma"]
    return pd.DataFrame({
        "source_dataset": "AWS 지점정보 (stn_inf.php, inf=AWS)",
        "provider": "기상청 API허브",
        "endpoint": "https://" + idx["source"],
        "request_method": "GET",
        "request_params_redacted": idx["params"].map(params_json),
        "source_schema_version": "고정폭 14열 STN_ID LON LAT STN_SP HT HT_WD LAU_ID STN_AD STN_KO STN_EN FCT_ID LAW_ID BASIN LAW_ADDR (파일 헤더 기준)",
        "requested_period_start": tm.map(kst),
        "requested_period_end": tm.map(kst),
        "collected_at": idx["collected_at"].astype(str),
        "collected_at_basis": "파일 저장 시각 (구조화할 때 인덱스 생성)",
        "bronze_file": "stn/" + idx["file"],
        "sha256": idx["sha256"],
        "size_bytes": idx["size_bytes"],
        "record_count": idx["n_rows"],
        "collection_status": "success",          # fetch_stn.py는 #7777END를 확인한 응답만 저장
        "error_code": "",
        "source_temporal_resolution": "시점 스냅샷 (매달 1일 00시, 2025-12-31 23시)",
        "source_spatial_resolution": "관측소 지점 (좌표, 법정동코드)",
        "license": lic, "license_url": url,
        "index_file": "stn_index.parquet",
    })


def adm():
    idx = pd.read_parquet(f"{BRONZE}/adm_index.parquet")
    chk = pd.read_csv(f"{BRONZE}/adm_raw_check.csv")
    rows_per_snapshot = chk.groupby("snapshot")["n_excel_rows"].sum()

    # zip 이름 → 푼 폴더 이름: 'jscode20260101(말소코드포함).zip' → 'jscode20260101_malso'
    def snapshot_of(name):
        return name.replace("(말소코드포함)", "_malso").replace(".zip", "") if name.endswith(".zip") else None

    snap = idx["file"].map(snapshot_of)
    unknown = [s for s in snap.dropna() if s not in rows_per_snapshot.index]
    assert not unknown, f"adm_raw_check.csv에 없는 폴더: {unknown}"
    base_date = idx["file"].map(lambda f: "20261026" if "20261026" in f else "20260101").map(ymd)
    lic, url = LICENSE["mois"]
    return pd.DataFrame({
        "source_dataset": "행정기관(행정동) 및 관할구역(법정동) 코드",
        "provider": "행정안전부 (누리집 게시판)",
        "endpoint": idx["source"],
        "request_method": "웹 게시판 내려받기",
        "request_params_redacted": "{}",
        "source_schema_version": idx["file"].map(
            lambda f: "엑셀 3개 (KIKcd_B, KIKcd_H, KIKmix)" if f.endswith(".zip") else "Layout 문서 (표 아님)"),
        "requested_period_start": base_date,
        "requested_period_end": base_date,
        "collected_at": idx["collected_at"].astype(str),
        "collected_at_basis": "파일 저장 시각 (내려받은 시각)",
        "bronze_file": "adm/" + idx["file"],
        "sha256": idx["sha256"],
        "size_bytes": idx["size_bytes"],
        "record_count": snap.map(rows_per_snapshot),   # hwpx는 비어 있음
        "collection_status": "success",
        "error_code": "",
        "source_temporal_resolution": "코드표 스냅샷 (기준일)",
        "source_spatial_resolution": "행정동·법정동 코드 (전국)",
        "license": lic, "license_url": url,
        "index_file": "adm_index.parquet",
    })


def boundary():
    """행정동 경계 zip: 인덱스가 따로 없어서 파일을 직접 읽어 만들고 boundary_index.parquet로도 저장한다."""
    import pyogrio
    base = f"{BRONZE}/boundary"
    ref_day = {"202411": "2024-11-30", "202512": "2025-12-31"}      # 경계 기준월의 말일 (silver_plan 경계 기준일과 같음)
    kst_tz = timezone(timedelta(hours=9))
    rows = []
    for ver in sorted(os.listdir(base)):
        folder = os.path.join(base, ver)
        if not os.path.isdir(folder):
            continue
        for z in sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip")):
            path = os.path.join(folder, z)
            with open(path, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            layers = sorted({os.path.basename(n)[:-4] for n in names if n.lower().endswith(".shp")})
            gemd = [n for n in names if n.endswith("TL_SCCO_GEMD.shp")]
            n_gemd = sum(pyogrio.read_info(f"zip://{path}!{g}")["features"] for g in gemd)
            rows.append({
                "version": ver, "file": f"{ver}/{z}", "sha256": sha, "size_bytes": os.path.getsize(path),
                "layers": ",".join(layers), "has_prj": any(n.lower().endswith(".prj") for n in names),
                "n_gemd": n_gemd,
                "collected_at": datetime.fromtimestamp(os.path.getmtime(path), kst_tz).isoformat(timespec="seconds"),
            })
    idx = pd.DataFrame(rows)
    idx.to_parquet(f"{BRONZE}/boundary_index.parquet", index=False)
    lic, url = LICENSE["mois"]
    return pd.DataFrame({
        "source_dataset": "행정구역 경계 (구역의 도형, 시도별 zip). Silver에서 쓰는 층은 TL_SCCO_GEMD (행정동)",
        "provider": "행정안전부 (주소기반산업지원서비스, 신청·승인 후 내려받기)",
        "endpoint": "https://business.juso.go.kr",
        "request_method": "웹 신청·승인 후 내려받기",
        "request_params_redacted": "{}",
        "source_schema_version": "SHP 층 " + idx["layers"] + " / 좌표계 정보(.prj) " + idx["has_prj"].map({True: "있음", False: "없음 → EPSG:5179로 확인"}) + " / 글자 cp949",
        "requested_period_start": idx["version"].map(ref_day),
        "requested_period_end": idx["version"].map(ref_day),
        "collected_at": idx["collected_at"],
        "collected_at_basis": "파일 수정 시각 (내려받은 시각)",
        "bronze_file": "boundary/" + idx["file"],
        "sha256": idx["sha256"],
        "size_bytes": idx["size_bytes"],
        "record_count": idx["n_gemd"],          # 행정동 층(TL_SCCO_GEMD) 구역 수. 출장소 포함
        "collection_status": "success",
        "error_code": "",
        "source_temporal_resolution": "경계 스냅샷 (기준월 2024-11, 2025-12)",
        "source_spatial_resolution": "행정동 경계 폴리곤 (EPSG:5179)",
        "license": lic, "license_url": url,
        "index_file": "boundary_index.parquet",
    })


def main():
    m = pd.concat([aws(), fire(), stn(), adm(), boundary()], ignore_index=True)[COLUMNS]
    m["record_count"] = m["record_count"].astype("Int64")
    m.to_parquet(f"{OUT}.parquet", index=False)
    m.to_csv(f"{OUT}.csv", index=False, encoding="utf-8-sig")

    print(f"manifest: {len(m):,}행 → {OUT}.parquet / .csv\n")
    s = m.groupby("index_file").agg(파일=("bronze_file", "size"), 건수=("record_count", "sum"),
                                    실패=("collection_status", lambda x: (x != "success").sum()))
    print(s.to_string())
    print("\n이용조건 미확인:", sorted(m.loc[m["license"] == "미확인", "index_file"].unique()))
    print("sha256 없는 행:", m["sha256"].isna().sum())


if __name__ == "__main__":
    main()