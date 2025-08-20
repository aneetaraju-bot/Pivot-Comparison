import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pivot Comparator", page_icon="📊", layout="wide")
st.title("📊 Generalized Pivot Comparator")

st.caption(
    "Upload two pivot CSVs (single pair) or many CSVs (batch). "
    "App aligns by keys, compares all shared headings, computes Change/%Change for numeric fields, "
    "flags Same/Different for text fields, and lets you focus on particular parameter(s)."
)

# -------------------- CACHING --------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(upfile) -> pd.DataFrame:
    df = pd.read_csv(upfile)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_resource(show_spinner=False)
def compile_regex(pattern: str):
    return re.compile(pattern)

# -------------------- HELPERS --------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    # convert "45%" and "1,234" safely
    if s.dtype == "object":
        s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="ignore")
    return s

def numericize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = coerce_numeric_series(df[c])
    return df

def guess_key_columns(df: pd.DataFrame, max_key_cols=5):
    """
    Heuristic: categorical or low-cardinality columns are likely keys.
    """
    n = max(len(df), 1)
    cands = []
    for c in df.columns:
        dt = df[c].dtype
        if dt == "object" or pd.api.types.is_categorical_dtype(dt):
            cands.append(c); continue
        try:
            uniq = df[c].nunique(dropna=False)
            if 1 < uniq <= min(50, max(3, n // 10)):
                cands.append(c)
        except Exception:
            pass
    return cands[:max_key_cols]

def safe_pct_change(curr, prev):
    if pd.isna(curr) and pd.isna(prev):
        return np.nan
    if prev in [None, 0] or pd.isna(prev):
        if pd.isna(curr) or curr in [None]:
            return np.nan
        if curr == 0:
            return 0.0
        return np.inf if curr > 0 else -np.inf
    return (curr - prev) / prev

def compare_pair(df1: pd.DataFrame, df2: pd.DataFrame, key_cols, tol=1e-12):
    """
    Align by keys and compare all shared columns.
    Returns: long-form comparison, wide numeric summary, keys used, compared columns.
    """
    df1 = numericize(normalize_cols(df1))
    df2 = numericize(normalize_cols(df2))

    shared_cols = sorted(list(set(df1.columns).intersection(set(df2.columns))))
    if not shared_cols:
        raise ValueError("No shared columns between the two files.")

    # ensure provided keys exist in shared; if none, fallback to row index
    key_cols = [k for k in key_cols if k in shared_cols]
    if not key_cols:
        df1["_row_idx_"] = range(len(df1))
        df2["_row_idx_"] = range(len(df2))
        key_cols = ["_row_idx_"]
        if "_row_idx_" not in shared_cols:
            shared_cols.append("_row_idx_")

    A = df1[shared_cols].copy()
    B = df2[shared_cols].copy()
    cmp_cols = [c for c in shared_cols if c not in key_cols]
    if not cmp_cols:
        raise ValueError("No comparable columns found (only keys shared).")

    # melt to long for consistent field-wise comparison
    A_long = A.melt(id_vars=key_cols, value_vars=cmp_cols, var_name="Field", value_name="Last")
    B_long = B.melt(id_vars=key_cols, value_vars=cmp_cols, var_name="Field", value_name="This")

    AB = pd.merge(A_long, B_long, on=key_cols + ["Field"], how="outer")

    # detect numeric per Field
    results = []
    for fld, grp in AB.groupby("Field", dropna=False, sort=False):
        sub = grp.copy()
        # treat field numeric if either side is numeric dtype
        is_num = pd.api.types.is_numeric_dtype(sub["Last"]) or pd.api.types.is_numeric_dtype(sub["This"])
        if is_num:
            sub["Last"] = pd.to_numeric(sub["Last"], errors="coerce")
            sub["This"] = pd.to_numeric(sub["This"], errors="coerce")
            sub["Change"] = sub["This"].fillna(0) - sub["Last"].fillna(0)
            sub["% Change"] = (sub.apply(lambda r: safe_pct_change(r["This"], r["Last"]), axis=1) * 100)

            def status_num(r):
                a = 0 if pd.isna(r["Last"]) else r["Last"]
                b = 0 if pd.isna(r["This"]) else r["This"]
                diff = b - a
                if abs(diff) <= tol:
                    return "Same"
                return "Up" if diff > 0 else "Down"

            sub["Status"] = sub.apply(status_num, axis=1)
            sub["Type"] = "numeric"
            results.append(sub[key_cols + ["Field", "Last", "This", "Change", "% Change", "Status", "Type"]])
        else:
            # text compare
            def status_txt(r):
                a, b = r["Last"], r["This"]
                if pd.isna(a) and pd.isna(b):
                    return "Same"
                return "Same" if str(a) == str(b) else "Different"

            sub["Change"] = np.nan
            sub["% Change"] = np.nan
            sub["Status"] = sub.apply(status_txt, axis=1)
            sub["Type"] = "text"
            results.append(sub[key_cols + ["Field", "Last", "This", "Change", "% Change", "Status", "Type"]])

    long_cmp = pd.concat(results, ignore_index=True)

    # build wide numeric summary
    num_only = long_cmp[long_cmp["Type"] == "numeric"].copy()
    if not num_only.empty:
        blocks = []
        for fld, g in num_only.groupby("Field", dropna=False, sort=False):
            b = g[key_cols + ["Last", "This", "Change", "% Change"]].copy()
            b = b.rename(columns={
                "Last": f"{fld} | Last",
                "This": f"{fld} | This",
                "Change": f"{fld} | Change",
                "% Change": f"{fld} | % Change",
            })
            blocks.append(b)
        wide = None
        for b in blocks:
            wide = b if wide is None else pd.merge(wide, b, on=key_cols, how="outer")
    else:
        wide = pd.DataFrame(columns=key_cols)

    return long_cmp, wide, key_cols, cmp_cols

def download_csv(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

def infer_numeric_candidates(df1: pd.DataFrame, df2: pd.DataFrame, shared: list) -> list:
    sample = pd.concat([df1[shared], df2[shared]], ignore_index=True)
    out = []
    for c in shared:
        s = sample[c]
        if s.dtype == "object":
            s = s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False)
            s = pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
    return out

def filter_selected_metrics(long_cmp_df: pd.DataFrame, key_cols: list, fields: list) -> pd.DataFrame:
    if not fields:
        return pd.DataFrame()
    sub = long_cmp_df[(long_cmp_df["Type"] == "numeric") & (long_cmp_df["Field"].isin(fields))].copy()
    cols = list(key_cols) + ["Field", "Last", "This", "Change", "% Change", "Status"]
    sub = sub[cols]
    sub = sub.sort_values(by=["Field", "% Change"], ascending=[True, False], na_position="last")
    return sub

# -------------------- UI --------------------
with st.sidebar:
    st.header("Upload mode")
    mode = st.radio("Select", ["Single pair (2 files)", "Batch pairs (many files)"])
    tol = st.number_input("Numeric equality tolerance", value=1e-12, step=1e-12, format="%.12f")
    st.caption("Two numeric cells within this absolute difference are counted as **Same**.")

# ---------- SINGLE PAIR ----------
if mode == "Single pair (2 files)":
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Pivot A (e.g., Last Week)", type=["csv"], key="a")
    with c2:
        f2 = st.file_uploader("Pivot B (e.g., This Week)", type=["csv"], key="b")

    if f1 and f2:
        df1 = read_csv_cached(f1)
        df2 = read_csv_cached(f2)

        st.expander("Preview A").dataframe(df1.head(20), use_container_width=True)
        st.expander("Preview B").dataframe(df2.head(20), use_container_width=True)

        shared = sorted(list(set(df1.columns) & set(df2.columns)))
        if not shared:
            st.error("No shared columns found.")
        else:
            # guess keys from merged sample
            sample = pd.concat([df1[shared], df2[shared]], ignore_index=True)
            guessed = [k for k in guess_key_columns(sample) if k in shared]
            key_cols = st.multiselect("Key columns", options=shared, default=guessed)

            try:
                long_cmp, wide_cmp, used_keys, compared_cols = compare_pair(df1, df2, key_cols, tol=tol)
                st.success(f"Compared {len(compared_cols)} shared fields across {len(used_keys)} key columns.")

                # ---- Parameter picker for Change % ----
                numeric_candidates = infer_numeric_candidates(df1, df2, shared)
                param_pick = st.multiselect(
                    "🎯 Pick metric(s) to show Change % for",
                    options=sorted([c for c in set(numeric_candidates) if c not in used_keys]),
                    default=[],
                    help="Choose one or more numeric headings (e.g., Batch Health, Avg Consumption)"
                )
                focused = filter_selected_metrics(long_cmp, used_keys, param_pick)

                # ---- Outputs ----
                st.subheader("Long-form comparison (numeric + text)")
                st.dataframe(long_cmp, use_container_width=True)
                download_csv(long_cmp, "⬇️ Download long comparison", "comparison_long.csv")

                st.subheader("Wide numeric summary (per metric)")
                if not wide_cmp.empty:
                    st.dataframe(wide_cmp, use_container_width=True)
                    download_csv(wide_cmp, "⬇️ Download wide numeric", "comparison_wide.csv")
                else:
                    st.info("No numeric fields to summarize.")

                st.subheader("📈 Selected metric(s) — Change % only")
                if not param_pick:
                    st.info("Select one or more metrics above to focus on their Change %.")
                elif focused.empty:
                    st.warning("No rows found for the selected metric(s).")
                else:
                    st.dataframe(focused, use_container_width=True)
                    download_csv(focused, "⬇️ Download selected metrics", "selected_metrics_change.csv")

            except Exception as e:
                st.error(f"Comparison failed: {e}")

# ---------- BATCH MODE ----------
else:
    files = st.file_uploader(
        "Upload multiple pivot CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="batch"
    )
    pair_regex = st.text_input(
        "Pairing regex (must define 'base' and 'period')",
        value=r"^(?P<base>.+)_(?P<period>w1|w2|last|this)\.csv$",
        help="Files with same 'base' and different 'period' form a pair. Example: teamA_w1.csv & teamA_w2.csv"
    )

    if files:
        rx = compile_regex(pair_regex)
        buckets = {}
        for f in files:
            m = rx.match(f.name)
            if not m:
                st.warning(f"Skipped (no regex match): {f.name}")
                continue
            base = m.group("base")
            period = m.group("period")
            try:
                df = read_csv_cached(f)
                buckets.setdefault(base, {})[period] = df
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")

        st.write(f"Detected groups: {len(buckets)}")
        all_long = []
        all_wide = []
        numeric_fields_union = set()

        for base, d in buckets.items():
            if len(d) < 2:
                st.warning(f"Skipping `{base}`: need two periods, got {list(d.keys())}")
                continue
            periods = sorted(list(d.keys()))
            p_last, p_this = periods[0], periods[1]
            df1, df2 = d[p_last], d[p_this]

            shared = sorted(list(set(df1.columns) & set(df2.columns)))
            if not shared:
                st.warning(f"`{base}` has no shared columns; skipped.")
                continue

            guessed = [k for k in guess_key_columns(pd.concat([df1[shared], df2[shared]], ignore_index=True)) if k in shared]
            try:
                long_cmp, wide_cmp, used_keys, compared_cols = compare_pair(df1, df2, guessed, tol=tol)
                long_cmp.insert(0, "PAIR", base)
                all_long.append(long_cmp)
                # collect numeric fields for dropdown
                numeric_fields_union |= set(long_cmp[long_cmp["Type"] == "numeric"]["Field"].unique())

                if not wide_cmp.empty:
                    wide_cmp.insert(0, "PAIR", base)
                    all_wide.append(wide_cmp)
            except Exception as e:
                st.warning(f"Pair `{base}` failed: {e}")

        if all_long:
            long_all = pd.concat(all_long, ignore_index=True)
            st.subheader("Batch: long-form comparison")
            st.dataframe(long_all, use_container_width=True)
            download_csv(long_all, "⬇️ Download ALL (long)", "batch_comparison_long.csv")

            # ---- Batch parameter filter ----
            st.subheader("🎯 Batch: filter by parameter (Change %)")
            param_choice = st.selectbox(
                "Pick one numeric parameter", 
                options=["<None>"] + sorted(numeric_fields_union)
            )
            if param_choice != "<None>":
                filt = long_all[(long_all["Type"] == "numeric") & (long_all["Field"] == param_choice)].copy()
                cols = ["PAIR"] + [c for c in filt.columns if c not in ["Type"]]
                filt = filt[cols].sort_values(by=["PAIR", "% Change"], ascending=[True, False], na_position="last")
                st.dataframe(filt, use_container_width=True)
                download_csv(filt, f"⬇️ Download batch ({param_choice})", f"batch_{param_choice}_change.csv")

        if all_wide:
            wide_all = pd.concat(all_wide, ignore_index=True, sort=False)
            st.subheader("Batch: wide numeric summary")
            st.dataframe(wide_all, use_container_width=True)
            download_csv(wide_all, "⬇️ Download ALL (wide)", "batch_comparison_wide.csv")

        if not all_long and not all_wide:
            st.info("No successful pairs. Check your regex and file naming.")
