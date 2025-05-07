## --------------------------------------------------------------------------- IMPORTS --------------------------------------------------------------------------- ##

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

## --------------------------------------------------------------------------- FUNCTIONS --------------------------------------------------------------------------- ##

def plot_scenario(scenario = "a"):
    """
    Plots the historical and future scenarios for the given scenario letter.
    If no scenario letter is provided, it will plot the default scenario.
    
    - Parameters:
    scenario_letter (str): The letter representing the scenario to plot. Default is a.
    """

    # Plot the features
    historical_data = pd.read_csv("forecasts/historical_data.csv", parse_dates=["Date"])

    # Standardize Revenue separately (only for plotting)
    revenue_scaler = StandardScaler()
    historical_data["Revenue"] = revenue_scaler.fit_transform(historical_data[["Total"]])

    # Define the columns to standardize
    columns_to_standardize = ["RESP", "EUA", "NG Price", "BESS"]  # Include "Revenue" for standardization

    # Initialize the scaler and fit it on the historical dataset
    scaler = StandardScaler()
    historical_data[columns_to_standardize] = scaler.fit_transform(historical_data[columns_to_standardize])

    # Load the future dataset
    future_data = pd.read_csv("forecasts/Future dataset with BESS.csv", parse_dates=["Date"])

    # Rename columns in future_dataset to match historical_data
    if scenario == None:
        column_mapping = {
            "Ra": "RESP",
            "Ea": "EUA",
            "Ga": "NG Price",
            "Ba": "BESS"
        }
    elif scenario == "a" or scenario == "b" or scenario == "c":
        column_mapping = {
            f"R{scenario}": "RESP",
            f"E{scenario}": "EUA",
            f"G{scenario}": "NG Price",
            f"B{scenario}": "BESS"
        }
    else:
        raise ValueError("Invalid scenario letter. Use 'a', 'b', or 'c'.")
        
    future_data.rename(columns=column_mapping, inplace=True)

    # Apply the same scaler (fitted on historical data) to the future dataset
    future_data[columns_to_standardize] = scaler.transform(future_data[columns_to_standardize])  # Exclude "Revenue" for future data

    # Plot the standardized historical and future data
    plt.figure(figsize=(20, 10))
    plt.plot(historical_data["Date"], historical_data["RESP"], label="RESP (Historical, Standardized)", linestyle='--', color='orange')
    plt.plot(historical_data["Date"], historical_data["EUA"], label="EUA (Historical, Standardized)", linestyle='--', color='blue')
    plt.plot(historical_data["Date"], historical_data["NG Price"], label="NG Price (Historical, Standardized)", linestyle='--', color='green')
    plt.plot(historical_data["Date"], historical_data["BESS"], label="BESS (Historical, Standardized)", linestyle='--', color='red')

    plt.plot(future_data["Date"], future_data["RESP"], label="RESP (Future, Standardized)", color='orange')
    plt.plot(future_data["Date"], future_data["EUA"], label="EUA (Future, Standardized)", color='blue')
    plt.plot(future_data["Date"], future_data["NG Price"], label="NG Price (Future, Standardized)", color='green')
    plt.plot(future_data["Date"], future_data["BESS"], label="BESS (Future, Standardized)", color='red')

    # Add historical standardized revenues as a black line
    plt.plot(historical_data["Date"], historical_data["Revenue"], label="Revenue (Historical, Standardized)", color='black', linewidth=2)

    plt.title("Historical and Future Scenarios (Standardized)")
    plt.xlabel("Date")
    plt.ylabel("Standardized features")
    plt.legend()
    plt.grid()
    plt.show()

    # Define smooth logistic curve that passes as close as possible to the midpoint
def smooth_logistic_growth(dates, start, end, steepness=4):
    """
    Generates a smooth logistic growth curve between two points (start and end) over a given date range.

    - Parameters:
        - dates (pd.Series): A pandas Series of dates.
        - start (float): The starting value of the curve.
        - end (float): The ending value of the curve.
    """
    t = (dates - dates[0]).days / 365.25
    duration = t[-1]
    x0 = duration / 2
    k = steepness / duration

    # Base logistic curve between start and end
    raw_logistic = 1 / (1 + np.exp(-k * (t - x0)))
    scaled_logistic = start + (end - start) * (raw_logistic - raw_logistic[0]) / (raw_logistic[-1] - raw_logistic[0])
    
    return scaled_logistic

## --------------------------------------------------------------------------- TRAINING FUNCTIONS --------------------------------------------------------------------------- ##

def train_model(feature_names=["RESP", "NG_Price"], outcome="Imbalance",
                            historical_data_path="processed_data/historical_data.csv", show_summary=True):
    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    historical_data.set_index("Date", inplace=True)
    historical_data["month"] = historical_data.index.month

    # # Monthly dummies (excluding the first to avoid multicollinearity)
    # month_dummies = pd.get_dummies(historical_data["month"], prefix="Month", drop_first=True)

    # Features and target
    # X = pd.concat([historical_data[feature_names], month_dummies], axis=1).astype(float)
    X = historical_data[feature_names].astype(float)
    y = historical_data[outcome]

    # Remove outliers
    z = np.abs(stats.zscore(y))
    threshold = 3
    outliers_y = np.where(z > threshold)[0]
    X = X.drop(X.index[outliers_y])
    y = y.drop(y.index[outliers_y])

    X_const = sm.add_constant(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(cv=tscv, store_cv_results=False, alphas=np.logspace(-2, 2, 50)).fit(X_const, y)

    # Best model metrics
    best_alpha = model.alpha_
    y_pred = model.predict(X_const)
    mae = mean_absolute_error(y, y_pred)

    if show_summary:
        print(f"✅ Best Alpha: {best_alpha}")
        print(f"✅ Mean Absolute Error on Full Data: {mae:.4f}")

    return model

def train_model_day_ahead(historical_data_path="processed_data/historical_data.csv", show_summary=True, include_lag=False):
    """
    Trains an OLS regression model using the specified features and outcome variable.
    - Parameters:
        - feature_names (list): List of feature names to include in the model.
        - outcome (str): The name of the outcome variable.
        - historical_data_path (str): Path to the historical dataset.
        - show_summary (bool): Whether to display the model summary.
    """
    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    historical_data.set_index("Date", inplace=True)
    historical_data["month"] = historical_data.index.month

    # Create monthly dummies (drop first month to avoid multicollinearity)
    month_dummies = pd.get_dummies(historical_data["month"], prefix="Month", drop_first=True)

    feature_names=["RESP", "NG_Price", "BESS"]

    # Combine features and month dummies
    X = pd.concat([historical_data[feature_names], month_dummies], axis=1)
    X = sm.add_constant(X).astype(float)

    y = historical_data["Day-ahead Only"]

    model = sm.OLS(y, X).fit(cov_type='HC2')

    if show_summary:
        print(model.summary())

    return model

## --------------------------------------------------------------------------- FORECASTING FUNCTIONS --------------------------------------------------------------------------- ##

def forecast_with_scenarios(model, outcome="Imbalance",
                            future_data_path="processed_data/Future dataset with BESS.csv",
                            historical_data_path="processed_data/historical_data.csv",
                            show_plot=True):
    """
    Forecasts future scenarios using the trained model and specified features, including monthly dummies.
    """
    future_data = pd.read_csv(future_data_path, parse_dates=["Date"])
    future_data["month"] = future_data["Date"].dt.month

    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    historical_data.set_index("Date", inplace=True)

    # # Prepare month dummies (must match training: months 2–12, drop January)
    # month_dummies = pd.get_dummies(future_data["month"], prefix="Month", drop_first=True)

    # Mapping future scenario columns to standardized features
    scenario_mapping = {
        "RESP": ["Ra", "Rb", "Rc", "Rd", "Re", "Rf"],
        "NG_Price": ["Ga", "Gb", "Gc"],
    }

    scenario_combinations = list(itertools.product(
        *[scenario_mapping[key] for key in scenario_mapping.keys()]
    ))

    future_combination_dfs = []
    for combination in scenario_combinations:
        combination_name = "_".join(combination)

        try:
            # Base features
            temp_df = pd.DataFrame({
                "RESP": future_data[combination[0]].values,
                "NG_Price": future_data[combination[1]].values,
            })

            # Combine with month dummies
            # X_future = pd.concat([temp_df, month_dummies], axis=1)
            X_future = temp_df.copy()

            # Add constant and ensure float type
            X_future = sm.add_constant(X_future).astype(float)

            # Predict revenue
            revenue_forecast = model.predict(X_future)

            # clipping values
            # if outcome == "Intraday":
            #     revenue_forecast = np.clip(revenue_forecast, -2000, 20000)
            # elif outcome == "Imbalance":
            #     revenue_forecast = np.clip(revenue_forecast, -10000, 40000)

            # Store forecast
            future_comb_df = pd.DataFrame({
                "Date": future_data["Date"],
                f"Total Revenue Forecast ({combination_name})": revenue_forecast
            })
            future_combination_dfs.append(future_comb_df.set_index("Date"))

        except KeyError as e:
            print(f"Missing column in combination {combination}: {e}")
            continue

    all_forecasts = pd.concat(future_combination_dfs, axis=1)

    if show_plot:
        plt.figure(figsize=(20, 5))
        plt.plot(historical_data.index, historical_data[outcome], label="Historical Revenue")
        for column in all_forecasts.columns:
            plt.plot(all_forecasts.index, all_forecasts[column], label=column)
        plt.title(f"Forecasts for {outcome} Revenue Across Scenarios")
        plt.grid(True)
        plt.show()

    return all_forecasts

def forecast_with_scenarios_rolling(model, outcome="Imbalance", feature_names=["RESP", "NG Price", "BESS"],
                                     future_data_path="Datasets/Processed Data/Future dataset with BESS.csv",
                                     historical_data_path="Datasets/Processed Data/historical_data.csv",
                                     include_lag=False, show_plot=True):
    
    if not include_lag:
        all_forecasts = forecast_with_scenarios(model, outcome, feature_names, future_data_path, historical_data_path, show_plot)
    else:
        future_data = pd.read_csv(future_data_path, parse_dates=["Date"])
        future_data["month"] = future_data["Date"].dt.month

        historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
        historical_data.set_index("Date", inplace=True)

        # Mapping future scenario columns to standardized features
        scenario_mapping = {
            "RESP": ["Ra", "Rb", "Rc", "Rd", "Re", "Rf"],
            "NG Price": ["Ga", "Gb", "Gc"],
            "BESS": ["Ba", "Bb", "Bc", "Bd", "Be", "Bf"]
        }

        # Generate all combinations of scenarios
        scenario_combinations = list(itertools.product(
            *[scenario_mapping[key] for key in scenario_mapping.keys()]
        ))

        # Create DataFrame for each scenario combination
        future_combination_dfs = []
        for combination in scenario_combinations:
            combination_name = "_".join(combination)

            try:
                forecast_values = []
                last_lag = historical_data["Day-ahead Only"].iloc[-1] if include_lag else None

                for t in range(len(future_data)):
                    row = {
                        "RESP": future_data.loc[t, combination[0]],
                        "NG Price": future_data.loc[t, combination[1]],
                        "BESS": future_data.loc[t, combination[2]]
                    }

                    row["Day Ahead Lag"] = last_lag

                    X_input = pd.DataFrame([row])
                    X_input["const"] = 1
                    X_input = X_input[["const"] + [col for col in X_input.columns if col != "const"]]

                    prediction = model.predict(X_input)[0]
                    forecast_values.append(prediction)

                    last_lag = prediction

                forecast_df = pd.DataFrame({
                    "Date": future_data["Date"],
                    f"Total Revenue Forecast ({combination_name})": forecast_values
                })
                future_combination_dfs.append(forecast_df.set_index("Date"))

            except KeyError as e:
                print(f"Missing column in combination {combination}: {e}")
                continue

        # Combine all scenario forecasts
        all_forecasts = pd.concat(future_combination_dfs, axis=1)

        # Plot results
        if show_plot:
            plt.figure(figsize=(20, 5))
            plt.plot(historical_data.index, historical_data[outcome], label="Historical Revenue")
            for column in all_forecasts.columns:
                plt.plot(all_forecasts.index, all_forecasts[column], label=column)
            plt.title("Rolling Forecast with Scenarios")
            plt.xlabel("Date")
            plt.ylabel("Forecasted Revenue")
            plt.grid(True)
            plt.show()

    return all_forecasts
