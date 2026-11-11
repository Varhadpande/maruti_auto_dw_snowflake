# streamlit_app.py
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict

import matplotlib.pyplot as plt  # ✅ Matplotlib for PNG charts

# Snowflake native Streamlit: use the active session
try:
    from snowflake.snowpark.context import get_active_session
    SESSION = get_active_session()
except Exception:
    SESSION = None

st.set_page_config(page_title="Maruti KPIs Dashboard", layout="wide")
st.title("🚗 Maruti Analytics KPIs — Tables + PNG Charts")
st.caption("Backlog dataset excluded for performance. Charts render as static Matplotlib PNG images.")

# ------------------------------------------------------------------------------
# Config: MARTS views (Backlog intentionally removed)
# ------------------------------------------------------------------------------
DB, SC = "MARUTI_AUTO_DWH", "MARTS"
MARTS: Dict[str, str] = {
    "forecast_vs_sales": f"{DB}.{SC}.V_FORECAST_VS_SALES_DAILY",
    # "backlog":         f"{DB}.{SC}.V_BACKLOG_INSIGHTS_DAILY",
    "cx":                f"{DB}.{SC}.V_CX_SCORE_DAILY",
    "warranty":          f"{DB}.{SC}.V_WARRANTY_DEFECTS_DAILY",
    "dealer":            f"{DB}.{SC}.V_DEALER_SCORECARD_DAILY",
    "inventory":         f"{DB}.{SC}.V_INVENTORY_FLOW_DAILY",
    "production":        f"{DB}.{SC}.V_PRODUCTION_KPIS_DAILY",
    "proc_lt":           f"{DB}.{SC}.V_PROCUREMENT_LT_DAILY",
}

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=300)
def fetch_view(name_key: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch a MARTS view into a Pandas DataFrame."""
    if SESSION is None:
        raise RuntimeError("No active Snowflake session. Run inside Snowflake or create a session.")

    full_name = MARTS[name_key]
    st.info(f"Fetching data from: `{full_name}`")  # ✅ show table/view name being queried

    sql = f"SELECT * FROM {full_name}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    try:
        df = SESSION.sql(sql).to_pandas()
    except Exception as e:
        st.error(f"{full_name} not available or failed to query:\n{e}")
        return pd.DataFrame()
    return df


def existing_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Return columns (original casing) that exist among candidates."""
    cset = {c.upper(): c for c in df.columns}
    found = []
    for c in candidates:
        cu = c.upper()
        if cu in cset:
            found.append(cset[cu])
    return found


def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    """Pick first available column among options."""
    cols = existing_cols(df, options)
    return cols[0] if cols else None


def coerce_date(df: pd.DataFrame) -> (pd.DataFrame, Optional[str]):
    """Try to coerce a date column if present; returns (df, date_col)."""
    if df.empty:
        return df, None
    date_candidates = [
        "DATE", "AS_OF_DATE",
        "INVOICE_DATE", "DELIVERY_DATE", "SERVICE_DATE",
        "BOOKING_DATE", "FORECAST_DATE", "GENERATED_DATE",
        "SNAPSHOT_DATE", "DISPATCH_DATE", "CLAIM_DATE",
        "PROD_DATE"
    ]
    date_col = pick_col(df, date_candidates) or next((c for c in df.columns if c.upper().endswith("_DATE")), None)
    if date_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df, date_col


def add_common_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Sidebar filters shared by tabs."""
    if df.empty:
        return df

    with st.sidebar.expander(f"Filters • {key_prefix}", expanded=False):
        # Date range
        df, date_col = coerce_date(df)
        if date_col:
            min_d, max_d = df[date_col].min(), df[date_col].max()
            if pd.notna(min_d) and pd.notna(max_d):
                d1, d2 = st.date_input(
                    "Date range",
                    value=(min_d.date(), max_d.date()),
                    key=f"{key_prefix}_date"
                )
                if d1 and d2:
                    df = df[(df[date_col] >= pd.Timestamp(d1)) & (df[date_col] <= pd.Timestamp(d2))]

        # Dimensions (all optional; only show if present)
        def multiselect_filter(colname: str, label: str):
            if colname and colname in df.columns:
                vals = sorted([v for v in df[colname].dropna().unique()])
                sel = st.multiselect(label, vals, key=f"{key_prefix}_{colname}")
                if sel:
                    return df[df[colname].isin(sel)]
            return df

        df = multiselect_filter(pick_col(df, ["REGION"]), "Region")
        df = multiselect_filter(pick_col(df, ["DEALER", "DEALER_NAME", "DEALER_CODE"]), "Dealer")
        df = multiselect_filter(pick_col(df, ["MODEL_NAME", "MODEL_CODE"]), "Model")
        df = multiselect_filter(pick_col(df, ["VARIANT_NAME", "VARIANT_CODE"]), "Variant")
        df = multiselect_filter(pick_col(df, ["FUEL_TYPE"]), "Fuel Type")
        df = multiselect_filter(pick_col(df, ["TRANSMISSION_TYPE"]), "Transmission")

    return df


def safe_group_mean(df: pd.DataFrame, by: str, value: str) -> pd.DataFrame:
    """Groupby mean if columns exist; else empty."""
    if by in df.columns and value in df.columns:
        g = df.groupby(by, dropna=False)[value].mean().reset_index()
        return g.sort_values(by)
    return pd.DataFrame()


def safe_group_sum(df: pd.DataFrame, by: str, values: List[str]) -> pd.DataFrame:
    """Groupby sum for available value columns."""
    cols = [v for v in values if v in df.columns]
    if by in df.columns and cols:
        g = df.groupby(by, dropna=False)[cols].sum().reset_index()
        return g.sort_values(by)
    return pd.DataFrame()


# ------------------------------------------------------------------------------
# Matplotlib chart helpers  (PNG output via st.pyplot)
# ------------------------------------------------------------------------------

def plot_line(df: pd.DataFrame, x: str, y_cols: List[str], title: str):
    y_use = [c for c in y_cols if c in df.columns]
    if not x or not y_use or df.empty:
        return
    df_plot = df[[x] + y_use].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in y_use:
        ax.plot(df_plot[x], df_plot[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    if x not in df.columns or y not in df.columns or df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_area(df: pd.DataFrame, x: str, y_cols: List[str], title: str):
    y_use = [c for c in y_cols if c in df.columns]
    if not x or not y_use or df.empty:
        return
    df_plot = df[[x] + y_use].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(df_plot[x], *[df_plot[c] for c in y_use], labels=y_use)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab_fc, tab_cx, tab_warr, tab_dealer, tab_inv, tab_prod, tab_proc = st.tabs([
    "Forecast vs Sales",
    "Customer Experience",
    "Warranty & Defects",
    "Dealer Scorecard",
    "Inventory Flow",
    "Production KPIs",
    "Procurement Lead Time",
])

# ------------------------------------------------------------------------------
# Forecast vs Sales
# ------------------------------------------------------------------------------
with tab_fc:
    df = fetch_view("forecast_vs_sales")
    if not df.empty:
        df = add_common_filters(df, "fc")
        df, date_col = coerce_date(df)
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Customer Experience
# ------------------------------------------------------------------------------
with tab_cx:
    df = fetch_view("cx")
    if not df.empty:
        df = add_common_filters(df, "cx")
        df, date_col = coerce_date(df)
        avg_cx = safe_group_mean(
            df,
            date_col or pick_col(df, ["DATE"]),
            pick_col(df, ["CX_SCORE", "NPS_SCORE", "SATISFACTION_SCORE"]) or ""
        )
        if not avg_cx.empty:
            plot_line(
                avg_cx,
                x=avg_cx.columns[0],
                y_cols=[avg_cx.columns[1]],
                title="Average CX / NPS over Time"
            )
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Warranty & Defects
# ------------------------------------------------------------------------------
with tab_warr:
    df = fetch_view("warranty")
    if not df.empty:
        df = add_common_filters(df, "warr")
        # Count by severity or fault system
        sev_col = pick_col(df, ["SEVERITY_LEVEL", "FAULT_SEVERITY", "SEVERITY"])
        cnt_col = pick_col(df, ["CLAIM_COUNT", "CLAIMS", "COUNT", "NUM_CLAIMS"]) or "___temp_count___"
        if cnt_col == "___temp_count___":
            df[cnt_col] = 1
        if sev_col:
            agg = df.groupby(sev_col, dropna=False)[cnt_col].sum().reset_index()
            plot_bar(agg, x=sev_col, y=cnt_col, title="Warranty Claims by Severity")
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Dealer Scorecard
# ------------------------------------------------------------------------------
with tab_dealer:
    df = fetch_view("dealer")
    if not df.empty:
        df = add_common_filters(df, "dealer")
        dealer_name = pick_col(df, ["DEALER", "DEALER_NAME", "DEALER_CODE"])
        score_col   = pick_col(df, ["DEALER_SCORE", "COMPOSITE_SCORE", "RATING", "SCORE"])
        if dealer_name and score_col:
            topn = (df[[dealer_name, score_col]]
                    .dropna()
                    .groupby(dealer_name, as_index=False)[score_col].mean()
                    .nlargest(15, score_col))
            plot_bar(topn, x=dealer_name, y=score_col, title="Top Dealers by Score (avg)")
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Inventory Flow
# ------------------------------------------------------------------------------
with tab_inv:
    df = fetch_view("inventory")
    if not df.empty:
        df = add_common_filters(df, "inv")
        df, date_col = coerce_date(df)
        plot_area(
            df.sort_values(date_col) if date_col else df,
            x=date_col or pick_col(df, ["DATE"]),
            y_cols=["OPENING_STOCK_UNITS", "UNITS_PRODUCED", "UNITS_DISPATCHED", "CLOSING_STOCK_UNITS"],
            title="Inventory Flow (Daily)"
        )
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Production KPIs
# ------------------------------------------------------------------------------
with tab_prod:
    df = fetch_view("production")
    if not df.empty:
        df = add_common_filters(df, "prod")
        df, date_col = coerce_date(df)
        plot_line(
            df.sort_values(date_col) if date_col else df,
            x=date_col or pick_col(df, ["DATE"]),
            y_cols=["UNITS_PRODUCED", "UNITS_REJECTED", "UNITS_REWORKED", "OEE_PERCENT"],
            title="Production KPIs (Daily)"
        )
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------------------
# Procurement Lead Time
# ------------------------------------------------------------------------------
with tab_proc:
    df = fetch_view("proc_lt")
    if not df.empty:
        df = add_common_filters(df, "proc")
        # Average lead time by vendor or material
        by_col = pick_col(df, ["VENDOR_NAME", "VENDOR_CODE", "MATERIAL_CODE", "MATERIAL_NAME"])
        lt_col = pick_col(df, ["LEAD_TIME_DAYS", "STANDARD_LEAD_TIME_DAYS", "AVG_LEAD_TIME_DAYS"])
        if by_col and lt_col:
            topn = (df[[by_col, lt_col]]
                    .dropna()
                    .groupby(by_col, as_index=False)[lt_col].mean()
                    .nlargest(20, lt_col))
            plot_bar(topn, x=by_col, y=lt_col, title="Avg Lead Time (Top 20)")
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=500)

st.caption("All charts are rendered via Matplotlib as static PNG images. If a chart is blank, that view may not expose the expected columns — the table still shows the raw data.")
