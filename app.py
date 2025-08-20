import streamlit as st
import pandas as pd

st.set_page_config(page_title="Pivot Compare (Fast)", layout="wide")
st.title("📊 Pivot Compare — Fast Deploy")

c1, c2 = st.columns(2)
with c1:
    f1 = st.file_uploader("Pivot A (CSV)", type=["csv"], key="a")
with c2:
    f2 = st.file_uploader("Pivot B (CSV)", type=["csv"], key="b")

@st.cache_data(show_spinner=False)
def load_csv(f):
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_num(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            s = out[c].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False)
            out[c] = pd.to_numeric(s, errors="ignore")
    return out

if f1 and f2:
    df1, df2 = load_csv(f1), load_csv(f2)
    shared = sorted(set(df1.columns) & set(df2.columns))
    if not shared:
        st.error("No shared columns between the two files.")
    else:
        # guess keys = shared object columns
        obj_guess = [c for c in shared if (df1[c].dtype == "object") or (df2[c].dtype == "object")]
        keys = st.multiselect("Key columns", shared, default=obj_guess[:3])

        if keys:
            df1, df2 = coerce_num(df1[shared]), coerce_num(df2[shared])
            A = df1.groupby(keys, dropna=False).sum(numeric_only=True).reset_index()
            B = df2.groupby(keys, dropna=False).sum(numeric_only=True).reset_index()
            merged = A.merge(B, on=keys, how="outer", suffixes=(" | Last"," | This"))

            for c in shared:
                if c in keys: 
                    continue
                last, this = f"{c} | Last", f"{c} | This"
                if last in merged and this in merged:
                    if pd.api.types.is_numeric_dtype(merged[last]) and pd.api.types.is_numeric_dtype(merged[this]):
                        merged[f"{c} | Change"] = merged[this] - merged[last]
                        denom = merged[last].replace(0, pd.NA)
                        merged[f"{c} | % Change"] = (merged[this] - merged[last]) / denom * 100

            st.dataframe(merged, use_container_width=True)
        else:
            st.info("Pick at least one key column.")
else:
    st.info("Upload two CSV pivot tables to begin.")
