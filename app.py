from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Superstore Dashboard", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent / "dataset" / "super_store.csv"


@st.cache_data
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='latin1')
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    return df


def format_money(value: float) -> str:
    return f"${value:,.2f}"


# ============================================================================
# Data Processing Functions
# ============================================================================


def filter_data(
    df: pd.DataFrame,
    selected_regions: list,
    selected_categories: list,
    selected_segments: list,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Apply filters to the dataset."""
    return df[
        (df["Region"].isin(selected_regions))
        & (df["Category"].isin(selected_categories))
        & (df["Segment"].isin(selected_segments))
        & (df["Order Date"].dt.date >= start_date)
        & (df["Order Date"].dt.date <= end_date)
    ]


def get_kpi_values(filtered: pd.DataFrame) -> dict:
    """Calculate KPI metrics."""
    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    profit_margin_pct = (total_profit / total_sales * 100) if total_sales else 0
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": filtered["Order ID"].nunique(),
        "total_quantity": filtered["Quantity"].sum(),
        "profit_margin_pct": profit_margin_pct,
    }


def get_sales_by_month(filtered: pd.DataFrame) -> pd.Series:
    """Get sales aggregated by month."""
    return (
        filtered.assign(Month=filtered["Order Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=True)["Sales"]
        .sum()
        .sort_index()
    )


def get_sales_by_category(filtered: pd.DataFrame) -> pd.Series:
    """Get sales aggregated by category."""
    return filtered.groupby("Category", as_index=True)["Sales"].sum().sort_values(ascending=False)


def get_profit_by_region(filtered: pd.DataFrame) -> pd.Series:
    """Get profit aggregated by region."""
    return filtered.groupby("Region", as_index=True)["Profit"].sum().sort_values(ascending=False)


def get_sales_by_segment(filtered: pd.DataFrame) -> pd.Series:
    """Get sales aggregated by segment."""
    return filtered.groupby("Segment", as_index=True)["Sales"].sum().sort_values(ascending=False)


def get_top_subcategories(filtered: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Get top N sub-categories by sales."""
    return (
        filtered.groupby("Sub-Category", as_index=True)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def get_profit_margin_by_category(filtered: pd.DataFrame) -> pd.DataFrame:
    """Calculate sales, profit, and profit margin percent by category."""
    summary = (
        filtered.groupby("Category", as_index=False)
        .agg({"Sales": "sum", "Profit": "sum"})
        .sort_values("Sales", ascending=False)
    )
    summary["Profit Margin %"] = (summary["Profit"] / summary["Sales"] * 100).fillna(0)
    return summary


def get_discount_impact(filtered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales/profit by discount bucket to show discount impact."""
    temp = filtered.copy()
    temp["Discount Bucket"] = pd.cut(
        temp["Discount"],
        bins=[-0.001, 0, 0.1, 0.2, 0.4, 0.6, 1.0],
        labels=["0%", "1-10%", "11-20%", "21-40%", "41-60%", "61-100%"],
    )

    impact = (
        temp.groupby("Discount Bucket", observed=False, as_index=False)
        .agg({"Sales": "sum", "Profit": "sum", "Order ID": "nunique"})
        .rename(columns={"Order ID": "Orders"})
    )
    impact["Profit Margin %"] = (impact["Profit"] / impact["Sales"] * 100).fillna(0)
    return impact


def get_top_and_bottom_products(filtered: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom products by profit."""
    product_profit = (
        filtered.groupby("Product Name", as_index=False)
        .agg({"Sales": "sum", "Profit": "sum", "Quantity": "sum"})
    )
    top_products = product_profit.sort_values("Profit", ascending=False).head(n)
    bottom_products = product_profit.sort_values("Profit", ascending=True).head(n)
    return top_products, bottom_products


def display_kpi_metrics(filtered: pd.DataFrame) -> None:
    """Display key performance indicator metrics."""
    kpis = get_kpi_values(filtered)
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Total Sales", format_money(kpis["total_sales"]))
    metric_col2.metric("Total Profit", format_money(kpis["total_profit"]))
    metric_col3.metric("Total Orders", f"{kpis['total_orders']:,}")
    metric_col4.metric("Items Sold", f"{int(kpis['total_quantity']):,}")
    metric_col5.metric("Profit Margin", f"{kpis['profit_margin_pct']:.2f}%")


def display_sales_trend(filtered: pd.DataFrame) -> None:
    """Display sales trend by month as a line chart."""
    sales_by_month = get_sales_by_month(filtered)
    st.subheader("Sales Trend by Month")
    st.line_chart(sales_by_month)


def display_sales_by_category(filtered: pd.DataFrame) -> None:
    """Display sales by category as a bar chart."""
    sales_by_category = get_sales_by_category(filtered)
    st.subheader("Sales by Category")
    st.bar_chart(sales_by_category)


def display_profit_by_region(filtered: pd.DataFrame) -> None:
    """Display profit by region as a bar chart."""
    profit_by_region = get_profit_by_region(filtered)
    st.subheader("Profit by Region")
    st.bar_chart(profit_by_region)


def display_top_subcategories(filtered: pd.DataFrame) -> None:
    """Display top 10 sub-categories by sales as a bar chart."""
    top_subcategories = get_top_subcategories(filtered)
    st.subheader("Top 10 Sub-Categories by Sales")
    st.bar_chart(top_subcategories)


def display_data_table(filtered: pd.DataFrame) -> None:
    """Display the filtered dataset as an interactive table."""
    st.subheader("Filtered Data")
    st.dataframe(filtered, use_container_width=True, height=320)
    st.caption(f"Showing {len(filtered):,} rows")


def display_pie_chart(data: pd.DataFrame, category_col: str, value_col: str, title: str) -> None:
    """Render a pie chart using Streamlit's Vega-Lite support."""
    chart_data = data.copy()
    total_value = chart_data[value_col].sum()
    chart_data["Percent"] = (chart_data[value_col] / total_value * 100).fillna(0)

    st.subheader(title)
    st.vega_lite_chart(
        chart_data,
        {
            "mark": {"type": "arc", "outerRadius": 120},
            "encoding": {
                "theta": {"field": value_col, "type": "quantitative"},
                "color": {"field": category_col, "type": "nominal"},
                "tooltip": [
                    {"field": category_col, "type": "nominal"},
                    {"field": value_col, "type": "quantitative", "format": ",.2f"},
                    {"field": "Percent", "type": "quantitative", "format": ".1f", "title": "Percent (%)"},
                ],
            },
            "view": {"stroke": None},
        },
        use_container_width=True,
    )

    legend_data = chart_data[[category_col, "Percent"]].copy()
    legend_data["Percent"] = legend_data["Percent"].map(lambda x: f"{x:.1f}%")
    st.dataframe(legend_data, use_container_width=True, hide_index=True)


def display_profitability_section(filtered: pd.DataFrame) -> None:
    """Display profitability-focused views."""
    st.subheader("Profitability by Category")
    profit_margin = get_profit_margin_by_category(filtered).set_index("Category")
    st.bar_chart(profit_margin[["Profit Margin %"]])
    st.dataframe(
        profit_margin.reset_index()[["Category", "Sales", "Profit", "Profit Margin %"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top and Bottom Products by Profit")
    top_products, bottom_products = get_top_and_bottom_products(filtered)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Top 10 Most Profitable Products**")
        st.dataframe(top_products[["Product Name", "Sales", "Profit", "Quantity"]], use_container_width=True, hide_index=True)
    with col_right:
        st.markdown("**Top 10 Least Profitable Products**")
        st.dataframe(bottom_products[["Product Name", "Sales", "Profit", "Quantity"]], use_container_width=True, hide_index=True)


def display_discount_impact_section(filtered: pd.DataFrame) -> None:
    """Display discount impact analysis."""
    st.subheader("Discount Impact on Profitability")
    impact = get_discount_impact(filtered)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Sales by Discount Bucket**")
        st.bar_chart(impact.set_index("Discount Bucket")[["Sales"]])
    with col_right:
        st.markdown("**Profit Margin by Discount Bucket (%)**")
        st.line_chart(impact.set_index("Discount Bucket")[["Profit Margin %"]])

    st.dataframe(
        impact[["Discount Bucket", "Sales", "Profit", "Orders", "Profit Margin %"]],
        use_container_width=True,
        hide_index=True,
    )


def display_download_section(filtered: pd.DataFrame) -> None:
    """Display download controls for filtered data."""
    st.subheader("Export")
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_superstore_data.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Superstore Sales Dashboard")
    st.caption("Simple interactive dashboard built with Streamlit from the Super Store dataset.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    df = df.dropna(subset=["Order Date"])

    st.sidebar.header("Filters")

    regions = sorted(df["Region"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())
    segments = sorted(df["Segment"].dropna().unique().tolist())

    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

    min_date = df["Order Date"].min().date()
    max_date = df["Order Date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    filtered = filter_data(
        df,
        selected_regions,
        selected_categories,
        selected_segments,
        start_date,
        end_date,
    )

    display_kpi_metrics(filtered)

    tab_overview, tab_profitability, tab_discount, tab_data = st.tabs(
        ["Overview", "Profitability", "Discount Analysis", "Data"]
    )

    with tab_overview:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            display_sales_trend(filtered)
        with chart_col2:
            display_sales_by_category(filtered)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            display_profit_by_region(filtered)
        with chart_col4:
            display_top_subcategories(filtered)

        pie_col1, pie_col2 = st.columns(2)
        with pie_col1:
            sales_category_df = get_sales_by_category(filtered).reset_index()
            sales_category_df.columns = ["Category", "Sales"]
            display_pie_chart(sales_category_df, "Category", "Sales", "Sales Share by Category")
        with pie_col2:
            sales_segment_df = get_sales_by_segment(filtered).reset_index()
            sales_segment_df.columns = ["Segment", "Sales"]
            display_pie_chart(sales_segment_df, "Segment", "Sales", "Sales Share by Segment")

    with tab_profitability:
        display_profitability_section(filtered)

    with tab_discount:
        display_discount_impact_section(filtered)

    with tab_data:
        display_data_table(filtered)
        st.divider()
        display_download_section(filtered)


if __name__ == "__main__":
    main()