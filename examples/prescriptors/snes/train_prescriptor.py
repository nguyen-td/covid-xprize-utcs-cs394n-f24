from copy import deepcopy

import neat
import numpy as np
import pandas as pd
import torch

from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger

from examples.predictors.lstm.xprize_predictor import XPrizePredictor
from examples.prescriptors.neat.utils import CASES_COL, PRED_CASES_COL, IP_COLS, IP_MAX_VALUES, get_predictions, prepare_historical_df

from network import Network
import matplotlib.pyplot as plt

import os

# Cutoff date for training data
CUTOFF_DATE = '2020-07-31'

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Number of days the prescriptors will look at in the past.
# Larger values here may make convergence slower, but give
# prescriptors more context. The number of inputs of each neat
# network will be NB_LOOKBACK_DAYS * (IP_COLS + 1).
NB_LOOKBACK_DAYS = 14

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
NB_EVAL_COUNTRIES = 20

# Load historical data with basic preprocessing
print("Loading historical data...")
df = prepare_historical_df()

# Restrict it to dates before the training cutoff
cutoff_date = pd.to_datetime(CUTOFF_DATE, format='%Y-%m-%d')
df = df[df['Date'] <= cutoff_date]

# As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
# so far as the geos for evaluation.
eval_geos = list(df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                ascending=False).head(NB_EVAL_COUNTRIES).index)
print("Nets will be evaluated on the following geos:", eval_geos)

# Pull out historical data for all geos
past_cases = {}
past_ips = {}
for geo in eval_geos:
    geo_df = df[df['GeoID'] == geo]
    past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
    past_ips[geo] = np.array(geo_df[IP_COLS])

# Gather values for scaling network output
ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

# Do any additional setup that is constant across evaluations
eval_start_date = pd.to_datetime(EVAL_START_DATE, format='%Y-%m-%d')
eval_end_date = pd.to_datetime(EVAL_END_DATE, format='%Y-%m-%d')

# Set up Neuroevolution problem
def fitness(network: torch.nn.Module):
    return -(new_cases * stringency)

covid_prescription = NEProblem(
    objective_sense='max',
    network=Network(num_inputs=NB_LOOKBACK_DAYS * 12 + NB_LOOKBACK_DAYS,  hidden_layers=[64], num_outputs=12),
    network_eval_func=fitness,
)

searcher = PGPE(
    covid_prescription,
    popsize=50,
    radius_init=2.25,
    center_learning_rate=0.2,
    stdev_learning_rate=0.1,
)
logger = PandasLogger(searcher)

# Start loop
df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
for ip_col in IP_COLS:
    df_dict[ip_col] = []

# Set initial data
eval_past_cases = deepcopy(past_cases)
eval_past_ips = deepcopy(past_ips)

# Make prescriptions one day at a time, feeding resulting
# predictions from the predictor back into the prescriptor.
for date in pd.date_range(eval_start_date, eval_end_date):
    date_str = date.strftime("%Y-%m-%d")

    # Prescribe for each geo
    for geo in eval_geos:

        # Prepare input data. Here we use log to place cases
        # on a reasonable scale; many other approaches are possible.
        X_cases = np.log(eval_past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
        X_ips = eval_past_ips[geo][-NB_LOOKBACK_DAYS:]
        X = np.concatenate([X_cases.flatten(),
                            X_ips.flatten()])

        # Get prescription
        trained_network = covid_prescription.parameterize_net(searcher.status["center"])
        prescribed_ips = trained_network(torch.tensor(X, dtype=torch.float32))

        # Map prescription to integer outputs
        prescribed_ips = (prescribed_ips.detach().numpy() * ip_max_values_arr).round()

        # Add it to prescription dictionary
        country_name, region_name = geo.split('__')
        if region_name == 'nan':
            region_name = np.nan
        df_dict['CountryName'].append(country_name)
        df_dict['RegionName'].append(region_name)
        df_dict['Date'].append(date_str)
        for ip_col, prescribed_ip in zip(IP_COLS, prescribed_ips):
            df_dict[ip_col].append(prescribed_ip)

    # Create dataframe from prescriptions.
    pres_df = pd.DataFrame(df_dict)
    # Make prediction given prescription for all countries
    pred_df = get_predictions(EVAL_START_DATE, date_str, pres_df)

    # Update past data with new day of prescriptions and predictions
    # ishann
    # Updated GeoID to not include RegionName since all NaNs.
    pres_df['GeoID'] = pres_df['CountryName'] + '__' #+ pres_df['RegionName'].astype(str)
    pred_df['GeoID'] = pred_df['CountryName'] + '__' #+ pred_df['RegionName'].astype(str)
    new_pres_df = pres_df[pres_df['Date'] == date_str]
    new_pred_df = pred_df[pred_df['Date'] == date_str]

    #ipdb.set_trace()

    for geo in eval_geos:
        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

        # Append array of prescriptions
        pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in IP_COLS]).reshape(1,-1)
        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

        # Append predicted cases
        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                            geo_pred[PRED_CASES_COL].values[0])
        
    new_cases = pred_df[PRED_CASES_COL].mean().mean()
    stringency = pres_df[IP_COLS].mean().mean()

    _ = StdOutLogger(searcher)
    searcher.run(50)
    # logger.to_dataframe().mean_eval.plot()
    # plt.show()