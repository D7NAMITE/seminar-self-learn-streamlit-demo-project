from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    # Kaggle exports are sometimes cp1252/latin1 instead of utf-8.
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unable to decode CSV using utf-8, cp1252, or latin1 encodings.")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    return df.dropna(subset=["Order Date", "Sales", "Profit", "Discount", "Quantity"])


def format_num(value: float) -> str:
    return f"{value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    st.set_page_config(
        page_title="Superstore Sales Dashboard",
        page_icon="📊",
        layout="wide",
    )

    data_path = Path("dataset") / "super_store.csv"
    if not data_path.exists():
        st.error("Dataset not found at dataset/super_store.csv")
        st.stop()

    df = load_data(data_path)

    st.title("Superstore Sales Dashboard")
    st.caption("Interactive exploration of sales, profit, discounts, and product performance.")

    with st.sidebar:
        st.header("Filters")

        min_date = df["Order Date"].min().date()
        max_date = df["Order Date"].max().date()
        selected_dates = st.date_input(
            "Order date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date, end_date = min_date, max_date

        regions = sorted(df["Region"].dropna().unique().tolist())
        selected_regions = st.multiselect("Region", options=regions, default=regions)

        categories = sorted(df["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Category", options=categories, default=categories)

    filtered_df = df[
        (df["Order Date"].dt.date >= start_date)
        & (df["Order Date"].dt.date <= end_date)
        & (df["Region"].isin(selected_regions))
        & (df["Category"].isin(selected_categories))
    ].copy()

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Please broaden your filters.")
        st.stop()

    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    profit_margin = total_profit / total_sales if total_sales else 0.0
    avg_discount = filtered_df["Discount"].mean()
    total_orders = filtered_df["Order ID"].nunique()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Sales", f"${format_num(total_sales)}")
    metric_cols[1].metric("Total Profit", f"${format_num(total_profit)}")
    metric_cols[2].metric("Profit Margin", format_percent(profit_margin))
    metric_cols[3].metric("Average Discount", format_percent(avg_discount))
    metric_cols[4].metric("Unique Orders", f"{total_orders:,}")

    filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=filtered_csv,
        file_name="filtered_superstore_data.csv",
        mime="text/csv",
    )

    tab_overview, tab_geo, tab_product = st.tabs(["Overview", "Geography", "Products"])

    with tab_overview:
        monthly = (
            filtered_df.set_index("Order Date")
            .resample("ME")[["Sales", "Profit"]]
            .sum()
            .rename_axis("Month")
            .reset_index()
        )

        st.subheader("Monthly Sales and Profit Trend")
        st.line_chart(monthly.set_index("Month")[["Sales", "Profit"]], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sales by Segment")
            sales_by_segment = filtered_df.groupby("Segment", as_index=True)["Sales"].sum().sort_values(ascending=False)
            st.bar_chart(sales_by_segment, use_container_width=True)

        with col2:
            st.subheader("Profit by Category")
            profit_by_category = (
                filtered_df.groupby("Category", as_index=True)["Profit"].sum().sort_values(ascending=False)
            )
            st.bar_chart(profit_by_category, use_container_width=True)

    with tab_geo:
        st.subheader("Sales by Region")
        region_sales = filtered_df.groupby("Region", as_index=True)["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(region_sales, use_container_width=True)

        st.subheader("Top 10 States by Sales")
        state_sales = filtered_df.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        st.dataframe(state_sales.head(10), use_container_width=True)

        st.subheader("Negative-Profit States (highlighted)")
        state_profit = filtered_df.groupby("State", as_index=False)["Profit"].sum().sort_values("Profit", ascending=True)
        st.dataframe(state_profit.head(15), use_container_width=True)

        st.subheader("Top 15 Cities by Profit")
        city_profit = filtered_df.groupby("City", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
        st.dataframe(city_profit.head(15), use_container_width=True)

    with tab_product:
        st.subheader("Top 15 Products by Sales")
        product_sales = (
            filtered_df.groupby("Product Name", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        )
        st.dataframe(product_sales.head(15), use_container_width=True)

        st.subheader("Bottom 15 Products by Profit")
        product_profit = (
            filtered_df.groupby("Product Name", as_index=False)["Profit"].sum().sort_values("Profit", ascending=True)
        )
        st.dataframe(product_profit.head(15), use_container_width=True)

        st.subheader("Sub-Category Performance")
        sub_category_perf = (
            filtered_df.groupby("Sub-Category", as_index=False)[["Sales", "Profit"]]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        st.dataframe(sub_category_perf, use_container_width=True)


if __name__ == "__main__":
    main()