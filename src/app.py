import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global CO2 Emissions Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Cleaning Function
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads the CO2 dataset and applies robust cleaning to remove aggregates.
    Verified cleaning logic from EDA notebook.
    """
    # Load raw data (Streamlit runs from root directory)
    file_path = "data/co2_emissions_kt_by_country.csv"
    df = pd.read_csv(file_path)

    # --- Robust Cleaning Logic (Verified in EDA Notebook) ---
    ignore_list = [
        'World', 'Arab World', 'Central Europe and the Baltics',
        'East Asia & Pacific', 'East Asia & Pacific (excluding high income)', 
        'East Asia & Pacific (IDA & IBRD countries)',
        'Europe & Central Asia', 'Europe & Central Asia (excluding high income)', 
        'Europe & Central Asia (IDA & IBRD countries)',
        'European Union', 'Euro area',
        'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)', 
        'Latin America & the Caribbean (IDA & IBRD countries)',
        'Middle East & North Africa', 'Middle East & North Africa (excluding high income)', 
        'Middle East & North Africa (IDA & IBRD countries)',
        'North America', 'South Asia', 'South Asia (IDA & IBRD)',
        'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)', 
        'Sub-Saharan Africa (IDA & IBRD countries)',
        'Africa Eastern and Southern', 'Africa Western and Central',
        'High income', 'Low & middle income', 'Low income', 'Lower middle income', 
        'Middle income', 'Upper middle income', 'OECD members',
        'IDA & IBRD total', 'IBRD only', 'IDA total', 'IDA blend', 'IDA only',
        'Heavily indebted poor countries (HIPC)', 
        'Least developed countries: UN classification',
        'Fragile and conflict affected situations', 
        'Early-demographic dividend', 'Late-demographic dividend', 
        'Pre-demographic dividend', 'Post-demographic dividend',
        'Small states', 'Pacific island small states', 'Caribbean small states', 
        'Other small states', 'Not classified'
    ]

    aggregate_codes = [
        'ARB', 'CSS', 'EAP', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU',
        'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDX', 'INX',
        'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA',
        'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST',
        'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN',
        'TSA', 'TSS', 'UMC', 'WLD'
    ]

    def looks_like_aggregate(name):
        patterns = ['income', 'dividend', 'IBRD', 'IDA', 'excluding']
        return any(p in name for p in patterns)

    df_clean = df[
        (~df['country_name'].isin(ignore_list)) &
        (~df['country_code'].isin(aggregate_codes)) &
        (~df['country_name'].apply(looks_like_aggregate))
    ].copy()
    
    return df_clean

# -----------------------------------------------------------------------------
# 3. Load Data with Error Handling
# -----------------------------------------------------------------------------
try:
    df = load_and_clean_data()
except FileNotFoundError:
    st.error("❌ Error: 'data/co2_emissions_kt_by_country.csv' not found.")
    st.info("Please ensure your project structure is correct:\n```\nglobal-co2-insight/\n├── data/\n│   └── co2_emissions_kt_by_country.csv\n└── src/\n    └── app.py\n```")
    st.stop()

# Get data ranges
min_year = int(df['year'].min())
max_year = int(df['year'].max())
all_countries = sorted(df['country_name'].unique())

# Get Top 10 emitters (for default selection)
df_latest = df[df['year'] == max_year]
top_10_default = df_latest.nlargest(10, 'value')['country_name'].tolist()

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Filters")

# Year Range Slider (like tb_dashboard.py)
year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (max_year - 20, max_year)  # Default: last 20 years
)

# Country Multi-select (like tb_dashboard.py)
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=top_10_default[:5],  # Default: Top 5
    help="Choose countries to display in the time series chart"
)

# Top N selector for bar chart
top_n = st.sidebar.selectbox(
    "Top N Countries (Bar Chart)",
    options=[5, 10, 15, 20],
    index=1  # Default: 10
)

# Chart type selector
chart_type = st.sidebar.radio(
    "Time Series Chart Type",
    ["Area", "Line"],
    index=0
)

# -----------------------------------------------------------------------------
# 5. Main Dashboard UI
# -----------------------------------------------------------------------------
st.title("🌍 Global CO2 Emissions Dashboard")
st.markdown(f"""
Interactive exploration of CO2 emissions across countries ({min_year}–{max_year}).  
**Data source:** World Bank (via Kaggle)
""")

# -----------------------------------------------------------------------------
# 6. Key Metrics Row
# -----------------------------------------------------------------------------
# Get latest year data for metrics
df_latest_year = df[df['year'] == year_range[1]]
total_emissions = df_latest_year['value'].sum()
top_emitter = df_latest_year.loc[df_latest_year['value'].idxmax()]

# Calculate YoY change
if year_range[1] > min_year:
    df_prev_year = df[df['year'] == year_range[1] - 1]
    prev_total = df_prev_year['value'].sum()
    yoy_change = ((total_emissions - prev_total) / prev_total) * 100
else:
    yoy_change = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Selected Period", f"{year_range[0]}–{year_range[1]}")
with col2:
    st.metric(
        f"🏭 Total Emissions ({year_range[1]})", 
        f"{total_emissions/1e6:.2f}M kt",
        delta=f"{yoy_change:+.1f}% YoY"
    )
with col3:
    st.metric("🥇 Top Emitter", top_emitter['country_name'])
with col4:
    st.metric("🌐 Countries Tracked", f"{df_latest_year['country_name'].nunique()}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. Charts Section
# -----------------------------------------------------------------------------
# Create two columns for charts
chart_col1, chart_col2 = st.columns([1.2, 1])

# -- LEFT: Time Series Chart --
with chart_col1:
    st.subheader(f"📈 CO2 Emissions Trend ({year_range[0]}–{year_range[1]})")
    
    if not selected_countries:
        st.warning("⚠️ Please select at least one country from the sidebar.")
    else:
        # Filter data
        df_trend = df[
            (df['country_name'].isin(selected_countries)) &
            (df['year'].between(year_range[0], year_range[1]))
        ]
        
        # Create chart based on selection
        if chart_type == "Area":
            fig_trend = px.area(
                df_trend,
                x='year',
                y='value',
                color='country_name',
                title=f"CO2 Emissions Over Time",
                labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        else:
            fig_trend = px.line(
                df_trend,
                x='year',
                y='value',
                color='country_name',
                title=f"CO2 Emissions Over Time",
                labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                template='plotly_white',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        
        fig_trend.update_layout(
            hovermode='x unified',
            legend_title='Country',
            height=450
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# -- RIGHT: Top N Bar Chart --
with chart_col2:
    st.subheader(f"🏆 Top {top_n} Emitters ({year_range[1]})")
    
    top_n_data = df_latest_year.nlargest(top_n, 'value')
    
    fig_bar = px.bar(
        top_n_data,
        x='value',
        y='country_name',
        orientation='h',
        text_auto='.2s',
        title=f"Top {top_n} Countries",
        labels={'value': 'CO2 Emissions (kt)', 'country_name': ''},
        template='plotly_white',
        color='value',
        color_continuous_scale='Reds'
    )
    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=450,
        coloraxis_showscale=False
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. Growth Analysis Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Growth Rate Analysis")

# Calculate growth rates
growth_col1, growth_col2 = st.columns(2)

with growth_col1:
    st.markdown(f"**Fastest Growing ({year_range[0]}→{year_range[1]})**")
    
    df_start = df[df['year'] == year_range[0]].set_index('country_name')['value']
    df_end = df[df['year'] == year_range[1]].set_index('country_name')['value']
    
    # Filter significant countries only (> 10,000 kt)
    significant = df_end[df_end > 10000].index
    growth_rate = ((df_end - df_start) / df_start * 100).loc[significant].dropna()
    
    top_5_growth = growth_rate.nlargest(5)
    
    for country, rate in top_5_growth.items():
        start_val = df_start.get(country, 0)
        end_val = df_end.get(country, 0)
        st.markdown(f"🔺 **{country}**: {rate:.1f}% ({start_val:,.0f} → {end_val:,.0f} kt)")

with growth_col2:
    st.markdown(f"**Biggest Decreases ({year_range[0]}→{year_range[1]})**")
    
    bottom_5_growth = growth_rate.nsmallest(5)
    
    for country, rate in bottom_5_growth.items():
        start_val = df_start.get(country, 0)
        end_val = df_end.get(country, 0)
        st.markdown(f"🔻 **{country}**: {rate:.1f}% ({start_val:,.0f} → {end_val:,.0f} kt)")

# -----------------------------------------------------------------------------
# 9. Data Preview
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📋 View Raw Data"):
    st.dataframe(
        df[df['year'] == year_range[1]][['country_name', 'country_code', 'year', 'value']]
        .sort_values('value', ascending=False)
        .head(50),
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# 10. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Built with ❤️ using Streamlit | Data: World Bank CO2 Emissions Dataset
</div>
""", unsafe_allow_html=True)