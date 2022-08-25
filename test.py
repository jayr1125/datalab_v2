import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from autots import AutoTS
from autots.tools.shaping import infer_frequency
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests

start = time.time()

st.set_page_config(layout="wide")

# Display FLNT logo
image = Image.open(r"C:\Users\jrsal\Pictures\flnt logo.png")
st.sidebar.image(image,
                 width=160)

# Display file uploader (adding space beneath the FLNT logo)
st.sidebar.write("")
st.sidebar.write("")
data1 = st.sidebar.file_uploader("",type=["csv", "xls", "xlsx"])

st.sidebar.write("---")

# Check for errors during upload
try:
    # Read dataset file uploader
    if data1 is not None:
        if data1.name.endswith(".csv"):
            data_df1 = pd.read_csv(data1)
        else:
            data_df1 = pd.read_excel(data1)

    # Impute missing values with mean
    data_df1 = data_df1.fillna(data_df1.mean())

    # For choosing features and targets
    data_df1_types = data_df1.dtypes.to_dict()

    # Choosing features and target for file 1
    targets1 = []
    for key, val in data_df1_types.items():
        if val != object:
            targets1.append(key)

    help_dependent = "Dependent variable is the effect. It is the value that you are trying to forecast"
    help_independent = "Independent variable is the cause. It is the value which may contribute to the forecast"
    chosen_target1 = st.sidebar.selectbox("Choose dependent variable",
                                          targets1,
                                          help=help_dependent)
    features1 = list(data_df1_types.keys())
    features1.remove(chosen_target1)
    chosen_date1 = st.sidebar.selectbox("Choose date column to use",
                                        features1)
    chosen_features1 = st.sidebar.multiselect("Choose independent variable(s) to use",
                                              features1,
                                              help=help_independent)

    st.sidebar.write("---")

    new_cols1 = chosen_features1.copy()
    new_cols1.append(chosen_target1)

    data_df1 = data_df1[new_cols1]

    # Preprocess data for experiment setup
    data_df1_series = data_df1.copy()

    # For descriptive stats
    data_df1_cols = data_df1.columns
    data_df1_shape = data_df1.shape

    data_df1_series[chosen_date1] = pd.to_datetime(data_df1_series[chosen_date1],
                                                   dayfirst=True)

    data_df1_series.set_index(data_df1_series[chosen_date1],
                              inplace=True)
    data_df1_series.drop(chosen_date1,
                         axis=1,
                         inplace=True)

    inferred_frequency = infer_frequency(data_df1_series)
    st.sidebar.write(f"Inferred Frequency of Dataset Uploaded: {inferred_frequency}")

    st.sidebar.write("---")

    # Maximum number of lags for Granger-causality Test
    granger_lag = st.sidebar.number_input("Max lag for Granger-causality test",
                                          step=1,
                                          value=5,
                                          help="Maximum number of lag to use for checking causality of two time series")

    # Create tabs for plots and statistics
    plot_tab, stat_tab, forecast_tab, prescriptive_tab = st.tabs(["Plots",
                                                                  "Statistics",
                                                                  "Forecast",
                                                                  "Prescriptive"])

    # Test for stationarity of time series data
    def test_stationarity(timeseries):
        # perform Dickey-Fuller test
        dftest = adfuller(timeseries,
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        if dfoutput['p-value'] > 0.05:
            return False
        else:
            return True

    def decompose(df, target_col):
        """

        :param df: input time series dataframe
        :param target_col: target column of dataframe
        :return: seasonal component, trend component, residual component (in this order)
        """
        dec = seasonal_decompose(df[target_col])
        return dec.seasonal, dec.trend, dec.resid

    seasonality, trend, residual = decompose(data_df1_series, chosen_target1)

    stationarity_data1 = test_stationarity(data_df1_series[chosen_target1])

    data_stats = data_df1_series[chosen_target1].describe()
    mean_data1 = round(data_stats['mean'], 2)
    median_data1 = round(data_stats['50%'], 2)
    std_data1 = round(data_stats['std'], 2)

    # Show correlation coefficient of the uploaded files (must be the same target name)
    corr = round(data_df1.corr()[chosen_target1][0:-1], 2)

    # For Granger causality test
    residual_filled = residual.fillna(residual.mean())

    with stat_tab:

        st.header("Descriptive Statistics")
        st.write("---")

        # Show descriptive statistics for file 1
        st.metric("No. of Variables",
                  data_df1_shape[1])
        st.metric("No. of Observations",
                  data_df1_shape[0])
        st.metric("Mean",
                  mean_data1)
        st.metric("Median",
                  median_data1)
        st.metric("Standard Deviation",
                  std_data1)
        # st.metric("Seasonality", seasonal_data1)
        help_stationary = "This tells whether the dataset has seasonality or trend. " \
                          "A dataset with trend or seasonality is not stationary"
        st.metric("Stationarity",
                  stationarity_data1,
                  help=help_stationary)

        st.write("---")

        st.write(f"Correlation of Dependent Variable to Independent Variables", corr)

        st.write("---")

        st.subheader("Granger Causality Test Results")

        for feature in data_df1_series.columns:
            for i in range(1, granger_lag+1):
                p_val = grangercausalitytests(pd.DataFrame(zip(data_df1_series[feature], residual_filled)),
                                              maxlag=granger_lag,
                                              verbose=False)[i][0]['ssr_ftest'][1]
                if p_val < 0.05:
                    if feature != chosen_target1:
                        st.write(
                            f"Knowing the values of {feature} is useful in predicting {chosen_target1} at lag {i}: {True}")
        else:
            st.write("No statistically significant causality was found")

    with plot_tab:
        st.subheader(f"Plots for {data1.name}")

        # Data 1 plot
        fig1 = go.Figure()
        fig1.add_trace(go.Line(name=data1.name,
                               x=data_df1_series.index,
                               y=data_df1_series[chosen_target1]))
        fig1.update_xaxes(gridcolor='grey')
        fig1.update_yaxes(gridcolor='grey')
        fig1.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=chosen_date1,
                           yaxis_title=chosen_target1,
                           title=f"{chosen_date1} vs. {chosen_target1}")

        st.plotly_chart(fig1,
                        use_container_width=True)

        # Seasonality plot
        fig2 = go.Figure()
        fig2.add_trace(go.Line(name="Seasonality",
                               x=seasonality.index,
                               y=seasonality))

        fig2.update_xaxes(gridcolor='grey')
        fig2.update_yaxes(gridcolor='grey')
        fig2.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=seasonality.index.name,
                           yaxis_title=seasonality.name,
                           title=f"Seasonal Component of {data1.name}")

        st.plotly_chart(fig2,
                        use_container_width=True)

        # Trend plot
        fig3 = go.Figure()
        fig3.add_trace(go.Line(name="Trend",
                               x=trend.index,
                               y=trend))

        fig3.update_xaxes(gridcolor='grey')
        fig3.update_yaxes(gridcolor='grey')
        fig3.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=trend.index.name,
                           yaxis_title=trend.name,
                           title=f"Trend Component of {data1.name}")

        st.plotly_chart(fig3,
                        use_container_width=True)

        # Residual plot
        fig4 = go.Figure()
        fig4.add_trace(go.Line(name="Residual",
                               x=residual.index,
                               y=residual))

        fig4.update_xaxes(gridcolor='grey')
        fig4.update_yaxes(gridcolor='grey')
        fig4.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=residual.index.name,
                           yaxis_title=residual.name,
                           title=f"Residual Component of {data1.name}")

        st.plotly_chart(fig4,
                        use_container_width=True)

        st.subheader("Cross Correlation Plots")

        # Cross correlation plots
        for feat in data_df1_series.columns:
            lag_user = st.number_input(f"Cross correlation lag/shift for {feat}",
                                       step=1,
                                       key=feat)

            # Manual mode for cross correlation and choosing lag/shift
            fig5 = go.Figure()
            fig5.add_trace(go.Line(name=chosen_target1,
                                   x=data_df1_series.index,
                                   y=data_df1_series[chosen_target1]))
            fig5.add_trace(go.Line(name=f"Shifted {feat}",
                                   x=data_df1_series.index,
                                   y=data_df1_series[feat].shift(periods=lag_user)))
            corr_user = data_df1_series[chosen_target1].corr(data_df1_series[feat].shift(periods=lag_user))
            fig5.update_layout(xaxis_title=chosen_date1,
                               yaxis_title="Data",
                               title=f"Data Correlation: {round(corr_user, 2)}")

            st.plotly_chart(fig5,
                            use_container_width=True)

    with forecast_tab:

        data1_slider = st.sidebar.number_input("Forecast Horizon",
                                               min_value=1,
                                               value=5,
                                               step=1)
        if st.button("Forecast"):
            model = AutoTS(
                forecast_length=data1_slider,
                frequency='infer',
                prediction_interval=0.95,
                ensemble=None,
                model_list='fast',
                max_generations=10,
                num_validations=2,
                no_negatives=True
            )

            model = model.fit(data_df1_series)
            model_name1 = model.best_model_name
            prediction = model.predict()

            x_data1 = prediction.forecast.index
            y_data1 = prediction.forecast[chosen_target1].values
            y_upper1 = prediction.upper_forecast[chosen_target1].values
            y_lower1 = prediction.lower_forecast[chosen_target1].values

            # Forecast 1 plot
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                name="Data",
                x=data_df1_series.index,
                y=data_df1_series[chosen_target1]
            ))

            fig5.add_trace(go.Scatter(
                name='Prediction',
                x=x_data1,
                y=y_data1,
                # mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))

            fig5.add_trace(go.Scatter(
                name='Upper Bound',
                x=x_data1,
                y=y_upper1,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))

            fig5.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_data1,
                y=y_lower1,
                marker=dict(color="#70B0E0"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

            fig5.update_xaxes(gridcolor='grey')
            fig5.update_yaxes(gridcolor='grey')
            fig5.update_layout(xaxis_title=chosen_date1,
                               yaxis_title=chosen_target1,
                               title=f"{data1.name} Forecast using {model_name1}",
                               hovermode="x",
                               colorway=["#7ee3c9"])

            st.plotly_chart(fig5,
                            use_container_width=True)

except (NameError, IndexError, KeyError) as e:
    pass

print("Done Rendering Application!")

end = time.time()
execution_time = end - start
st.write(f"Execution Time: {execution_time} seconds")
