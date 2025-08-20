import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="General Pivot Comparator", page_icon="📊", layout="wide")
st.title("📊 Generalized Pivot Comparator")

st.markdown("""
Upload **two pivot CSVs** with the **same/similar column headings**, or upload **multiple CSVs** and the app will
batch-compare files in **pairs** (e.g., `sales_A_w1.csv` vs `sales_A_w2.csv`, `sales_B_w1.csv` vs `sales_B_w2.csv`).

**What it does**
- Auto-detects likely **key columns** (row identifiers) but lets you override.
- Aligns rows by keys; compares **all shared columns** (numeric and non-numeric).
- For numeric columns: computes **Change**, **% Change**, and **Status** (Up/Down/Same).
- For non-numeric columns: flags **Same / Different** and shows both values.
- Exports long-form “tidy” comparison and a wide summary per metric.
""")

# ---------- Utilities ----------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="ignore")
    return series

def numericize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = coerce_numeric(df[c])
    return df

def guess_key_columns(df: pd.DataFrame, max_key_cols=5):
    """
    Heuristic:
    - object or categorical columns are likely keys
    - also include columns with low cardinality ratio
    - cap to first N
    """
    candidates = []
    n = max(len(df), 1)
    for c in df.columns:
        dtype = df[c].dtype
        if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
            candidates.append(c)
        else:
            # if low unique relative to rows, might be categorical
            try:
                uniq = df[c].nunique(dropna=False)
                if 1 < uniq <= min(50, max(3, n // 10)):  # small cardinality
                    candidates.append(c)
            except Exception:
                pass
    # Keep original order, cap size
    return candidates[:max_key_cols]

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
    Returns: long-form comparison and a wide summary (per metric).
    """
    # Standardize
    df1 = numericize(normalize_cols(df1))
    df2 = numericize(normalize_cols(df2))

    # Restrict to shared columns for robust merge
    shared_cols = sorted(list(set(df1.columns).intersection(set(df2.columns))))
    if not shared_cols:
        raise ValueError("No shared columns between the two files.")

    # Ensure keys are within shared
    key_cols = [k for k in key_cols if k in shared_cols]
    if not key_cols:
        # fall back to exact index-based merge if nothing to key on (rare for pivots)
        df1["_row_idx_"] = range(len(df1))
        df2["_row_idx_"] = range(len(df2))
        key_cols = ["_row_idx_"]
        if "_row_idx_" not in shared_cols:
            shared_cols = shared_cols + ["_row_idx_"]

    # Prepare reduced frames
    A = df1[shared_cols].copy()
    B = df2[shared_cols].copy()
    A["__src__"] = "Last"
    B["__src__"] = "This"

    # Identify compare columns = shared minus keys
    cmp_cols = [c for c in shared_cols if c not in key_cols]

    # If nothing to compare (e.g., only keys exist), bail
    if not cmp_cols:
        raise ValueError("No comparable columns found (only keys are shared).")

    # Melt to long to simplify column-wise comparison
    A_long = A.melt(id_vars=key_cols + ["__src__"], value_vars=cmp_cols, var_name="Field", value_name="Value")
    B_long = B.melt(id_vars=key_cols + ["__src__"], value_vars=cmp_cols, var_name="Field", value_name="Value")

    # Pivot back to two columns (Last, This)
    AB = pd.merge(
        A_long.drop(columns=["__src__"]).rename(columns={"Value": "Last"}),
        B_long.drop(columns=["__src__"]).rename(columns={"Value": "This"}),
        on=key_cols + ["Field"],
        how="outer"
    )

    # Determine numeric vs non-numeric per Field by inspecting available values
    def is_numeric_field(sub):
        # if any of Last/This is numeric => treat numeric
        return pd.api.types.is_numeric_dtype(sub["Last"]) or pd.api.types.is_numeric_dtype(sub["This"])

    results = []
    for field, grp in AB.groupby("Field", dropna=False, sort=False):
        sub = grp.copy()
        # numeric compare
        if is_numeric_field(sub):
            # cast to numeric safely
            sub["Last"] = pd.to_numeric(sub["Last"], errors="coerce")
            sub["This"] = pd.to_numeric(sub["This"], errors="coerce")
            sub["Change"] = (sub["This"].fillna(0) - sub["Last"].fillna(0))
            sub["% Change"] = sub.apply(lambda r: safe_pct_change(r["This"], r["Last"]), axis=1) * 100

            def status_num(r):
                a, b = r["Last"], r["This"]
                if pd.isna(a) and pd.isna(b): return "Same"
                a = 0 if pd.isna(a) else a
                b = 0 if pd.isna(b) else b
                diff = b - a
                if abs(diff) <= tol: return "Same"
                return "Up" if diff > 0 else "Down"

            sub["Status"] = sub.apply(status_num, axis=1)
            sub["Type"] = "numeric"
            results.append(sub[key_cols + ["Field", "Last", "This", "Change", "% Change", "Status", "Type"]])
        else:
            # text compare
            def status_txt(r):
                a, b = r["Last"], r["This"]
                if pd.isna(a) and pd.isna(b): return "Same"
                return "Same" if str(a) == str(b) else "Different"

            sub["Change"] = np.nan
            sub["% Change"] = np.nan
            sub["Status"] = sub.apply(status_txt, axis=1)
            sub["Type"] = "text"
            results.append(sub[key_cols + ["Field", "Last", "This", "Change", "% Change", "Status", "Type"]])

    long_cmp = pd.concat(results, ignore_index=True)

    # Wide numeric summary: for each numeric Field, bring back Last/This/Change/%Change side-by-side
    numeric_only = long_cmp[long_cmp["Type"] == "numeric"].copy()
    if not numeric_only.empty:
        wide_blocks = []
        for fld, g in numeric_only.groupby("Field", dropna=False, sort=False):
            block = g[key_cols + ["Last", "This", "Change", "% Change"]].copy()
            # add suffixes per field
            block = block.rename(columns={
                "Last": f"{fld} | Last",
                "This": f"{fld} | This",
                "Change": f"{fld} | Change",
                "% Change": f"{fld} | % Change",
            })
            wide_blocks.append(block)
        # progressive outer merges on keys
        wide = None
        for b in wide_blocks:
            wide = b if wide is None else pd.merge(wide, b, on=key_cols, how="outer")
    else:
        wide = pd.DataFrame(columns=key_cols)

    return long_cmp, wide, key_cols, cmp_cols

def download_csv(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

# ---------- UI ----------

with st.sidebar:
    st.header("Input mode")
    mode = st.radio("Choose", ["Single pair (2 files)", "Batch pairs (many files)"])
    tol = st.number_input("Numeric equality tolerance", value=1e-12, step=1e-12, format="%.12f")
    st.caption("Two numeric cells within this absolute difference are treated as **Same**.")

# --- Single Pair ---
if mode == "Single pair (2 files)":
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Pivot A (e.g., Last Week)", type=["csv"], key="a")
    with c2:
        f2 = st.file_uploader("Pivot B (e.g., This Week)", type=["csv"], key="b")

    if f1 and f2:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
        df1n = normalize_cols(df1)
        df2n = normalize_cols(df2)

        st.expander("Preview A").dataframe(df1n.head(20), use_container_width=True)
        st.expander("Preview B").dataframe(df2n.head(20), use_container_width=True)

        # guess keys from intersection
        shared_cols = sorted(list(set(df1n.columns).intersection(set(df2n.columns))))
        guessed = [k for k in guess_key_columns(pd.concat([df1n[shared_cols], df2n[shared_cols]], ignore_index=True)) if k in shared_cols]
        key_cols = st.multiselect("Key (row identifier) columns", options=shared_cols, default=guessed)

        # compare
        try:
            long_cmp, wide_cmp, used_keys, compared_cols = compare_pair(df1n, df2n, key_cols, tol=tol)
            st.success(f"Compared {len(compared_cols)} shared fields across {len(used_keys)} key columns.")

            st.subheader("Long-form comparison (works for all field types)")
            st.dataframe(long_cmp, use_container_width=True)
            download_csv(long_cmp, "⬇️ Download long comparison", "comparison_long.csv")

            st.subheader("Wide numeric summary (per metric)")
            if not wide_cmp.empty:
                st.dataframe(wide_cmp, use_container_width=True)
                download_csv(wide_cmp, "⬇️ Download wide numeric summary", "comparison_wide.csv")
            else:
                st.info("No numeric fields to summarize.")

            # Quick movers chart (numeric only)
            num = long_cmp[long_cmp["Type"] == "numeric"].copy()
            num = num.replace([np.inf, -np.inf], np.nan).dropna(subset=["% Change"])
            if not num.empty:
                st.subheader("Top movers by % Change (numeric)")
                topN = st.slider("How many?", min_value=5, max_value=50, value=10)
                label_key = st.selectbox("Label by (one key column)", used_keys, index=0 if used_keys else None)
                by_field = st.selectbox("Filter to a Field (optional)", ["<All>"] + sorted(num["Field"].dropna().unique().tolist()))
                plot_df = num.copy()
                if by_field != "<All>":
                    plot_df = plot_df[plot_df["Field"] == by_field]
                plot_df = plot_df.sort_values("% Change", ascending=False).head(topN)
                if not plot_df.empty:
                    fig = px.bar(plot_df, x=label_key, y="% Change", color="Field",
                                 hover_data=used_keys + ["Field", "Last", "This", "Change"])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No plottable data after filtering.")
        except Exception as e:
            st.error(f"Comparison failed: {e}")

# --- Batch Pairs ---
else:
    files = st.file_uploader("Upload many CSVs (we will auto-pair)", type=["csv"], accept_multiple_files=True, key="batch")
    st.caption("Tip: name files like `teamA_w1.csv` + `teamA_w2.csv`, `sales_East_last.csv` + `sales_East_this.csv`…")
    pair_hint = st.text_input("Pairing regex (2 groups = pair id + period id)", value=r"^(?P<base>.+)_(?P<period>w1|w2|last|this)\.csv$", help="We use the 'base' group to match pairs; 'period' must have two values.")

    if files:
        # Build index: {base: {period: dataframe}}
        compiled = re.compile(pair_hint)
        buckets = {}
        for f in files:
            m = compiled.match(f.name)
            if not m:
                st.warning(f"Skipped (no regex match): {f.name}")
                continue
            base = m.group("base")
            period = m.group("period")
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")
                continue
            buckets.setdefault(base, {})[period] = df

        st.write(f"Detected groups: {len(buckets)}")
        overall_long = []
        overall_wide = []

        for base, d in buckets.items():
            if len(d) < 2:
                st.warning(f"Skipping `{base}`: need exactly 2 periods, got {list(d.keys())}")
                continue

            # pick any two periods deterministically
            periods = sorted(list(d.keys()))
            p_last, p_this = periods[0], periods[1]
            df1, df2 = d[p_last], d[p_this]

            # shared cols for this pair
            shared = sorted(list(set(df1.columns).intersection(set(df2.columns))))
            if not shared:
                st.warning(f"`{base}` has no shared columns; skipped.")
                continue

            # guess keys for this pair
            guessed = [k for k in guess_key_columns(pd.concat([normalize_cols(df1)[shared], normalize_cols(df2)[shared]], ignore_index=True)) if k in shared]
            try:
                long_cmp, wide_cmp, used_keys, compared_cols = compare_pair(df1, df2, guessed, tol=tol)
                long_cmp.insert(0, "PAIR", base)
                wide_cmp.insert(0, "PAIR", base)
                overall_long.append(long_cmp)
                overall_wide.append(wide_cmp)
            except Exception as e:
                st.warning(f"Pair `{base}` failed: {e}")

        if overall_long:
            long_all = pd.concat(overall_long, ignore_index=True)
            st.subheader("Batch: long-form comparison")
            st.dataframe(long_all, use_container_width=True)
            download_csv(long_all, "⬇️ Download ALL (long)", "batch_comparison_long.csv")
        if overall_wide:
            wide_all = pd.concat(overall_wide, ignore_index=True, sort=False)
            st.subheader("Batch: wide numeric summary")
            st.dataframe(wide_all, use_container_width=True)
            download_csv(wide_all, "⬇️ Download ALL (wide)", "batch_comparison_wide.csv")
        if not overall_long and not overall_wide:
            st.info("No successful pairs to show yet. Check your regex and that each base has two CSVs.")
