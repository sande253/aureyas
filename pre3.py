import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO

# --- Page Setup ---
st.set_page_config(page_title="🔮 Trend Forecasting with Prophet", layout="wide")
st.title("🔮 Google Trends Forecasting Dashboard")
st.markdown("""
Use this app to explore and forecast keyword trends using [Facebook Prophet](https://facebook.github.io/prophet/), a time-series forecasting model developed by Meta.
""")


# --- Upload CSV ---
st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your `final_data.csv`", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=["date", "keyword"]).reset_index(drop=True)

        keywords = df['keyword'].unique()

        # --- Config Panel ---
        st.sidebar.header("⚙️ Forecast Settings")
        selected_keywords = st.sidebar.multiselect("Select keywords to forecast", keywords, default=keywords[:1])
        periods = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
        yearly = st.sidebar.checkbox("Add yearly seasonality", value=True)
        weekly = st.sidebar.checkbox("Add weekly seasonality", value=True)
        daily = st.sidebar.checkbox("Add daily seasonality", value=False)
        changepoint_prior_scale = st.sidebar.slider("Changepoint sensitivity", 0.01, 0.8, 0.1)

        st.markdown("### 📊 Forecast Results")

        for kw in selected_keywords:
            st.subheader(f"🔍 Forecast for: `{kw}`")
            keyword_df = df[df['keyword'] == kw]
            prophet_df = keyword_df[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})

            if len(prophet_df) < 10:
                st.warning(f"⚠ Not enough data for `{kw}`.")
                continue

            try:
                # Initialize Prophet
                model = Prophet(
                    yearly_seasonality=yearly,
                    weekly_seasonality=weekly,
                    daily_seasonality=daily,
                    changepoint_prior_scale=changepoint_prior_scale
                )
                model.fit(prophet_df)

                # Forecast
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)

                # Plot Forecast
                fig = plot_plotly(model, forecast)
                st.plotly_chart(fig, use_container_width=True)

                # Show raw data
                with st.expander("🔢 View forecast table"):
                    display_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
                    st.dataframe(display_df)

                    # Download CSV
                    csv = display_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download forecast CSV", csv, file_name=f"{kw}_forecast.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Failed to forecast `{kw}`: {e}")

    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")
else:
    st.info("👆 Upload a CSV with columns: `date`, `keyword`, and `value` to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using [Prophet](https://facebook.github.io/prophet/) and [Streamlit](https://streamlit.io)")
