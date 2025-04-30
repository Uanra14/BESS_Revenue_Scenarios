# BESS_Revenue_Scenarios

## Phase 1. Imbalance Forecasting - Gaussian Sampling

30/04/2025 - The original data from the EMS is at 15min resolution, for the purpose of efficiency and because of important hourly patterns in the imbalance market the sampling will be done at an hourly resolution. That is, we create a model for every hour of the day, and then we simulate data based on the model. For this purpose, processed_data contains now imbalance_revenue_hourly.csv, which contains hourly data from 2021 Jan to 2024 Dec.
