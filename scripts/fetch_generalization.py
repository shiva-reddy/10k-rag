"""Fetch most recent 10-K for 3 non-MAG7 filers (generalization test).

BRK.A (Berkshire Hathaway) — conglomerate, table-heavy
JPM   (JPMorgan Chase)     — bank, has industry-specific Items 7A
DDOG  (Datadog)            — tech mid-cap baseline

Writes HTML files + manifest.json into ~/ai-eng-datasets/non-mag7-10k/
"""
import json, os, time, urllib.request

UA = "Sanctum-Coursework shiva@example.com"
DEST = os.path.expanduser("~/ai-eng-datasets/non-mag7-10k")
os.makedirs(DEST, exist_ok=True)

FILERS = [
    ("BRKA", "0001067983", "Berkshire Hathaway"),
    ("JPM",  "0000019617", "JPMorgan Chase"),
    ("DDOG", "0001561550", "Datadog"),
]


def fetch_submissions_json(cik):
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    blocks = [d["filings"]["recent"]]
    for f in d["filings"].get("files", []):
        url2 = f"https://data.sec.gov/submissions/{f['name']}"
        req2 = urllib.request.Request(url2, headers={"User-Agent": UA})
        with urllib.request.urlopen(req2) as r:
            blocks.append(json.load(r))
        time.sleep(0.2)
    filings = []
    for block in blocks:
        for i, form in enumerate(block["form"]):
            if form == "10-K":
                filings.append({
                    "fy_end": block["reportDate"][i],
                    "filed": block["filingDate"][i],
                    "accession": block["accessionNumber"][i],
                    "doc": block["primaryDocument"][i],
                })
    filings.sort(key=lambda x: x["fy_end"], reverse=True)
    return filings


def download(cik, ticker, filing):
    cik_int = str(int(cik))
    acc_nodash = filing["accession"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{filing['doc']}"
    out = os.path.join(DEST, f"{ticker}-10k-{filing['fy_end']}.htm")
    if os.path.exists(out):
        return None, os.path.getsize(out)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    with open(out, "wb") as o:
        o.write(data)
    return url, len(data)


manifest = []
for ticker, cik, name in FILERS:
    print(f"\n=== {ticker} ({name}) ===")
    try:
        filings = fetch_submissions_json(cik)
        if not filings:
            print("  no 10-K found")
            continue
        f = filings[0]
        url, size = download(cik, ticker, f)
        status = "skip (exists)" if url is None else f"GET {size:,} bytes"
        print(f"  {f['fy_end']}  filed {f['filed']}  {status}")
        manifest.append({
            "ticker": ticker, "company": name, "cik": cik,
            "fy_end": f["fy_end"], "filed": f["filed"],
            "accession": f["accession"], "doc": f["doc"],
            "local_path": f"{ticker}-10k-{f['fy_end']}.htm",
            "size_bytes": size,
        })
        if url is not None:
            time.sleep(0.4)
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(0.3)

with open(os.path.join(DEST, "manifest.json"), "w") as o:
    json.dump(manifest, o, indent=2)
print(f"\nDone. Manifest written. {len(manifest)} filings.")
