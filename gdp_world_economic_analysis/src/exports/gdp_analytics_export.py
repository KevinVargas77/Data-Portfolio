import pandas as pd
import numpy as np
from datetime import datetime, timezone

def build_gdp_analytics_export(df_input, data_version=None, local_tz="America/Costa_Rica"):
    if df_input is None or df_input.empty:
        raise ValueError("Input DataFrame is empty.")
    
    df = df_input.copy()
    
    # Basic processing - find key columns
    c_country = None
    c_year = None
    value_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['country', 'pais'] and c_country is None:
            c_country = col
        elif col_lower in ['year'] and c_year is None:
            c_year = col
        elif col_lower in ['gdp_millions_usd', 'value', 'gdp'] and value_col is None:
            value_col = col
    
    if not c_country:
        raise ValueError("Country column required")
    if not c_year:
        raise ValueError("Year column required")
    if not value_col:
        raise ValueError("GDP value column required")
    
    # Convert GDP to billions if in millions
    if 'millions' in value_col.lower():
        df["Value"] = pd.to_numeric(df[value_col], errors="coerce") / 1000.0
    else:
        df["Value"] = pd.to_numeric(df[value_col], errors="coerce")
    
    # Add required columns
    df["Indicator"] = "GDP (Billions USD)"
    df["date"] = pd.to_datetime(df[c_year].astype(str) + "-12-31", errors="coerce")
    
    # Add Type column based on year
    current_year = datetime.now().year
    df["Type"] = np.where(df[c_year] <= current_year, "actual", "forecast")
    
    # Add analytics columns
    df = df.sort_values([c_country, c_year])
    df["YoY_Growth"] = df.groupby(c_country)["Value"].pct_change()
    
    # Simple derived metrics
    df["Horizon_Years"] = df[c_year] - current_year
    max_year_country = df.groupby(c_country)[c_year].transform("max")
    df["Latest_Year_Flag"] = (df[c_year] == max_year_country).astype(int)
    
    # Share of world GDP
    total_by_year = df.groupby(c_year)["Value"].transform("sum")
    df["Share_of_World_GDP"] = df["Value"] / total_by_year
    
    # Add traceability
    now_utc = datetime.now(timezone.utc)
    df["Run_TS"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["Data_Version"] = data_version or now_utc.strftime("%Y.%m.%d-001")
    df["Last_Actual_Year"] = current_year
    
    # Optional columns
    df["value_lo"] = np.nan
    df["value_hi"] = np.nan
    df["Model"] = np.nan
    
    # Column ordering
    ordered_cols = [
        c_country, c_year, "date", "Indicator", "Value", "Type", 
        "value_lo", "value_hi", "Model", "Run_TS", "Data_Version",
        "YoY_Growth", "Horizon_Years", "Last_Actual_Year", 
        "Latest_Year_Flag", "Share_of_World_GDP"
    ]
    
    # Keep only columns that exist
    final_cols = [col for col in ordered_cols if col in df.columns]
    df_out = df[final_cols].copy()
    
    # Generate filename and CSV bytes
    ts_local = now_utc.strftime("%Y%m%d_%H%M%SZ")
    file_name = f"gdp_world_analytics_{ts_local}.csv"
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    
    return df_out, file_name, csv_bytes


def prepare_streamlit_dataframe(df_raw, years, selected_countries=None):
    if selected_countries:
        df_filtered = df_raw[df_raw['Country'].isin(selected_countries)].copy()
    else:
        df_filtered = df_raw.copy()
    
    df_long = df_filtered.melt(
        id_vars=['Country'],
        value_vars=years,
        var_name='Year',
        value_name='GDP_Millions_USD'
    )
    
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['Year', 'GDP_Millions_USD'])
    
    current_year = datetime.now().year
    df_long['Type'] = np.where(df_long['Year'] <= current_year, 'actual', 'forecast')
    
    return df_long