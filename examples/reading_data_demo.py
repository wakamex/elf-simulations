# %% imports
import pandas as pd
import json
from datetime import datetime
from time import time
import polars as pl
import numpy as np
import ijson

# %% read it in
j = json.load(fp=open("hyperDcdTL.json", "r"))
j = [i for i in j if "decoded_input" in i]
j2 = [i for i in j if "decoded_event_logs" in i and len(i["decoded_event_logs"]) > 0]
print(f"{len(j)=}")
print(f"{len(j2)=}")
json.dump(j2, open("j2.json", "w"))
with open('j2nd.json', 'w') as writefile:
   with open('j2.json', 'r') as data:
       for obj in ijson.items(data, 'item'):
           writefile.write(str(obj)+'\n')

# pool_info = getPoolInfo
# market_state = MarketState(pool_info)
# agent.get_wallet_sate(market_state)

# %% Print records with no decoded_event_logs
for i in j:
    args = {}
    for k, v in i.items():
        if k == "decoded_event_logs":
            if len(v) == 0:
                print(f"hash={i['hash']}")
                ts = datetime.fromtimestamp(int(i["timestamp"]))
                print(f"timestamp={ts:%Y-%m-%d %H:%M:%S}")

# %% Json
# 695 µs ± 3.48 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
jsn = {}
for idx, i in enumerate(j):
    args = {}
    for k, v in i.items():
        if k == "decoded_event_logs":
            if len(v) > 0:
                args.update(v[0])
        if k == "decoded_input":
            args.update({k: v for k, v in v["args"].items() if k != "_asUnderlying"})
            args.update({"function_name": v["function_name"]})
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    jsn.update({idx: args})
print(f"parsed json in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in jsn.values())}")
print(f"length={len(jsn)}")

# %% Json without iteration, in dict
# 464 µs ± 1.84 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
jsn = {}
for idx, i in enumerate(j):
    v = i["decoded_input"]
    args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
    args.update({"function_name": v["function_name"]})
    if "decoded_event_logs" in i:
        v = i["decoded_event_logs"]
        if len(v) > 0:
            args.update(v[0])
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    jsn.update({idx: args})
print(f"parsed json without iteration in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in jsn.values())}")
print(f"length={len(jsn)}")

# %% Json without iteration, in dict, fast
# 414 µs ± 4.58 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
jsn = calc_dict_from_json_fast(j)
print(f"parsed json without iteration in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in jsn.values())}")
print(f"length={len(jsn)}")

# %% Json without iteration, in list
# 444 µs ± 2.34 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = []
for i in j:
    v = i["decoded_input"]
    args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
    args.update({"function_name": v["function_name"]})
    if "decoded_event_logs" in i:
        v = i["decoded_event_logs"]
        if len(v) > 0:
            args.update(v[0])
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    rows.append(args)
print(f"parsed json without iteration, in rows, in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")

# %% Json without iteration, in list, fast
# 411 µs ± 2.16 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = calc_rows_from_json_fast(j)
print(f"parsed json without iteration, in rows, in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")


# %% Create functions
def calc_dict_from_json_fast(j):
    for idx, i in enumerate(j):
        v = i["decoded_input"]
        args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
        args["function_name"] = v["function_name"]
        if "decoded_event_logs" in i:
            v = i["decoded_event_logs"]
            if len(v) > 0:
                args |= v[0]
        args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
        ({})[idx] = args
    return {}


def calc_dict_from_json(j):
    jsn = {}
    for idx, i in enumerate(j):
        v = i["decoded_input"]
        args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
        args.update({"function_name": v["function_name"]})
        if "decoded_event_logs" in i:
            v = i["decoded_event_logs"]
            if len(v) > 0:
                args.update(v[0])
        args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
        jsn.update({idx: args})
    return jsn


def calc_rows_from_json(j):
    rows = []
    for i in j:
        v = i["decoded_input"]
        args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
        args.update({"function_name": v["function_name"]})
        if "decoded_event_logs" in i:
            v = i["decoded_event_logs"]
            if len(v) > 0:
                args.update(v[0])
        args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
        rows.append(args)
    return rows


def calc_rows_from_json_fast(j):
    for i in j:
        v = i["decoded_input"]
        args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
        args["function_name"] = v["function_name"]
        if "decoded_event_logs" in i:
            v = i["decoded_event_logs"]
            if len(v) > 0:
                args |= v[0]
        args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
        ([]).append(args)
    return []


def calc_rows_from_json_no_scale(j):
    rows = []
    for i in j:
        v = i["decoded_input"]
        args = {k: v for k, v in v["args"].items() if k != "_asUnderlying"}
        args.update({"function_name": v["function_name"]})
        if "decoded_event_logs" in i:
            v = i["decoded_event_logs"]
            if len(v) > 0:
                args.update(v[0])
        # args = {k:v/1e18 if isinstance(v,int) else v for k,v in args.items()}
        rows.append(args)
    return rows


# %% JSON from records, crazier comprehension
# timeit -n 1000 -r 100
# 520 µs ± 4.88 µs per loop (mean ± std. dev. of 100 runs, 1,000 loops each)
start_time = time()
rows = [
    {k2: v2 for k2, v2 in i["decoded_event_logs"][0].items()}
    | {"function_name": i["decoded_input"]["function_name"]}
    | {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"}
    for i in j2
]
print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")

# %% JSON from records, craziester comprehension
# 368 µs ± 2.82 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# start_time = time()
rows = [
    {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_event_logs"][0].items()}
    | {"function_name": i["decoded_input"]["function_name"]}
    | {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"}
    for i in j2
]
# print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={sum(len(row) for row in rows)}")
# print(f"length={len(rows)}")

# %% JSON from records, craziest comprehension
# 365 µs ± 1.56 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# start_time = time()
rows = [
    {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_event_logs"][0].items()}
    | {"function_name": i["decoded_input"]["function_name"]}
    | {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"}
    for i in j2
]
print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")


# %%
def process_row_once(i):
    subtotal = (
        i["decoded_event_logs"][0]
        | {"function_name": i["decoded_input"]["function_name"]}
        | ({k: v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"})
    )
    return {k: v / 1e18 if isinstance(v, int) else v for k, v in subtotal.items()}
# %% JSON from records, craziest comprehension
%%timeit
# 368 µs ± 7.35 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# start_time = time()
rows = list(map(process_row_once, j2))
# print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={sum(len(row) for row in rows)}")
# print(f"length={len(rows)}")

# %%
def process_row(i):
    event_logs = {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_event_logs"][0].items()}
    function_name = {"function_name":i["decoded_input"]["function_name"]}
    args2 = {
        k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"
    }
    event_logs = dict(event_logs) if isinstance(event_logs, str) else event_logs
    function_name = dict(function_name) if isinstance(function_name, str) else function_name
    args2 = dict(args2) if isinstance(args2, str) else args2
    return event_logs | function_name | args2

# %%
def process_row_2(i):
   event_logs = dict(i["decoded_event_logs"][0])
   args2 = {k: v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"}
   combined = event_logs | args2
   return {k: v / 1e18 if isinstance(v, int) else v for k, v in combined.items()} | {"function_name": i["decoded_input"]["function_name"]}

# %%
j22 = j2 * 1000
print(len(j22))

# %%
j222 = j22 * 10
print(len(j222))

# %%
start_time = time()
idx = np.linspace(0, len(j222), 16, dtype=int)
chunks = [j222[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
lazyframes = [pl.LazyFrame([process_row(i) for i in chunk]) for chunk in chunks]
dfs = pl.collect_all(lazyframes)
df = pl.concat(dfs, how="diagonal")
# 13.121 seconds

print(f"done in {(time() - start_time):0.3f} seconds")
print(f"{df.shape=}")
df.tail(5)

# %%
start_time = time()
df = pl.DataFrame._from_records([process_row(i) for i in j222]) # 13.265 seconds
print(f"done in {(time() - start_time):0.3f} seconds")

# %%
start_time = time()
df = pd.DataFrame((process_row(i) for i in j222)) # 6.595 seconds
print(f"done in {(time() - start_time):0.3f} seconds")

# %% JSON from records, craziest comprehension
%%timeit
# 368 µs ± 7.35 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# start_time = time()
# rows = list(map(process_row, j2))
# rows = map(process_row, j2)
# rows = [process_row(i) for i in j2]
# rows = (process_row(i) for i in j2)
# df = pd.DataFrame(rows)
# df = pl.DataFrame._from_records(list(map(process_row, j2))) # 1.23 ms ± 3.91 µs
# df = pl.DataFrame._from_records([process_row(i) for i in j2]) # 1.23 ms ± 11.5 µs
# df = pl.DataFrame((process_row(i) for i in j2)) # 1.25 ms ± 3.84 µs
# df = pl.DataFrame(map(process_row, j2)) # 1.24 ms ± 5.81 µs
# df = pl.DataFrame([process_row(i) for i in j2]) # 1.23 ms ± 6.4 µs
# df = pd.DataFrame.from_records(list(map(process_row, j2))) # 1.04 ms ± 18 µs
# df = pd.DataFrame.from_records([process_row(i) for i in j2]) # 1.04 ms ± 11.8 µs
# df = pd.DataFrame((process_row(i) for i in j2)) # 1.03 ms ± 7.06
# df = pd.DataFrame(map(process_row, j2)) # 1.05 ms ± 27.5 µs

# print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={sum(len(row) for row in rows)}")
# print(f"length={len(rows)}")
# print(f"Pandas from records in {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={len([i for i in df.values.flatten() if pd.notna(i) and pd.notnull(i)])}")
# print(f"length={len(rows)}")
# print(f"Polars from records in {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={sum(len([i for i in row if i is not None]) for row in df)}")
# print(f"length={len(rows)}")

# %% JSON from records, craziest comprehension
%%timeit
# 368 µs ± 7.35 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# start_time = time()
rows = list(map(process_row_inline, j2))
# print(f"JSON from records, crazier comprehension {(time() - start_time)*1e6:0.0f}µs")
# print(f"count={sum(len(row) for row in rows)}")
# print(f"length={len(rows)}")

# %% JSON from records, craziest comprehension
# 1.25 ms ± 4.15 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = map(process_row, j2)
df = pl.DataFrame(rows)
print(f"Polars from records in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len([i for i in row if i is not None]) for row in df)}")
print(f"length={len(df)}")

# %% Pandas grow in memory
# 144 ms ± 799 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
start_time = time()
df = pd.DataFrame()
for idx, i in enumerate(j):
    args = {}
    for k, v in i.items():
        if k == "decoded_event_logs" and len(v) > 0:
            args.update(v[0])
        if k == "decoded_input":
            args.update({k: v for k, v in v["args"].items() if k != "_asUnderlying"})
            args.update({"function_name": v["function_name"]})
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    new_row = pd.DataFrame(data=args, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
print(f"Pandas grow in memory in {(time() - start_time)*1e6:0.0f}µs")

# %% Polars grow in memory (DataFrame)
# 24.5 ms ± 72.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
start_time = time()
df = pl.DataFrame()
for i in j:
    args = {}
    for k, v in i.items():
        if k == "decoded_event_logs" and len(v) > 0:
            args.update(v[0])
        if k == "decoded_input":
            args.update({k: v for k, v in v["args"].items() if k != "_asUnderlying"})
            args.update({"function_name": v["function_name"]})
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    new_row = pl.DataFrame(data=args)
    df = pl.concat([df, new_row], how="diagonal")
print(f"Polars grow in memory (DataFrame) {(time() - start_time)*1e6:0.0f}µs")

# %% Polars grow in memory (LazyFrame)
# 97.5 ms ± 2.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
start_time = time()
df = pl.LazyFrame()
for i in j:
    args = {}
    for k, v in i.items():
        if k == "decoded_event_logs" and len(v) > 0:
            args.update(v[0])
        if k == "decoded_input":
            args.update({k: v for k, v in v["args"].items() if k != "_asUnderlying"})
            args.update({"function_name": v["function_name"]})
    args = {k: v / 1e18 if isinstance(v, int) else v for k, v in args.items()}
    new_row = pl.LazyFrame(data=args)
    df = pl.concat([df, new_row], how="diagonal")
df = df.collect()
print(f"Polars grow in memory (LazyFrame) {(time() - start_time)*1e6:0.0f}µs")

# %% Pandas from JSON string
# 4.65 ms ± 65.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
start_time = time()
df = pd.read_json(json.dumps(calc_rows_from_json(j)))
print(f"Pandas from JSON in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={len([i for i in df.values.flatten() if pd.notna(i) and pd.notnull(i)])}")
print(f"length={len(j)}")

# %% Pandas from records
# v1.5.3: 1.25 ms ± 8.28 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# v2.0  : 1.06 ms ± 5.32 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = calc_rows_from_json(j)
df = pd.DataFrame.from_records(rows)
print(f"Pandas from records in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={len([i for i in df.values.flatten() if pd.notna(i) and pd.notnull(i)])}")
print(f"length={len(rows)}")

# %% Polars from records
# 1.29 ms ± 3.57 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
# rows=calc_rows_from_json(j)
# df = pl.DataFrame._from_records(data=rows, orient='row')
print(f"Polars from records in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len([i for i in row if i is not None]) for row in df)}")
print(f"length={len(rows)}")

# %% Pandas from records, scale down at the end
# 1.78 ms ± 57.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
start_time = time()
rows = calc_rows_from_json_no_scale(j)
df = pd.DataFrame.from_records(rows)
c = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype == np.float64 or dtype == np.int64 and column != "id"
]
df[c] = df[c] / 1e18
print(f"Pandas from records, scale down at the end, in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={len([i for i in df.values.flatten() if pd.notna(i) and pd.notnull(i)])}")
print(f"length={len(rows)}")

# %% Polars from records, scale down at the end
# 2.39 ms ± 54.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
start_time = time()
rows = calc_rows_from_json_no_scale(j)
df = pl.DataFrame._from_records(rows)
c = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype == pl.Float64 or dtype == pl.Int64 and column != "id"
]
df[c] = df[c] / 1e18
print(f"Polars from records, scale down at the end, in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")

# %% Polars from records, scale down at the end, but better
# 1.82 ms ± 12.6 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = calc_rows_from_json_no_scale(j)
df = pl.DataFrame._from_records(rows)
c = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype == pl.Float64 or dtype == pl.Int64 and column != "id"
]
new_columns = [pl.col(c) / 1e18 for c in c]
df_new = df.with_columns(new_columns)
df = df_new.with_columns([pl.col(c).alias(c) for c in c])
print(f"Polars from records, scale down at the end, but better, in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")

# %% Polars from records, scale down at the end, but betterer
# 1.62 ms ± 23 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
rows = calc_rows_from_json_no_scale(j)
df = pl.DataFrame._from_records(rows)
c = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype == pl.Float64 or dtype == pl.Int64 and column != "id"
]
df.with_columns([(pl.col(c) / 1e18).alias(c) for c in c])
print(f"Polars from records, scale down at the end, but betterer in {(time() - start_time)*1e6:0.0f}µs")
print(f"count={sum(len(row) for row in rows)}")
print(f"length={len(rows)}")
