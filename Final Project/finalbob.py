import quandl
import pandas as pd
import numpy as np

from yahoo_finance import Share
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import seaborn as sns

quandl.ApiConfig.api_key='qTvzicdBxcoUs5GsExRz'
universe = quandl.get_table('ZACKS/FC')
new_universe = universe.iloc[:][['ticker','per_end_date','basic_net_eps','ebitda','ebit','invty','tot_curr_asset','gross_prop_plant_equip','curr_portion_debt','tot_change_asset_liab','incr_decr_cash','net_acq_divst','change_income_tax','cost_good_sold','gross_profit','tot_deprec_amort','tot_revnu','net_tot_equity_issued_repurch','fgn_exchange_rate_adj','stock_based_compsn','wavg_shares_out','debt_issue_retire_net_tot','invst_activity_other','net_change_intang_asset','change_acct_rcv','tot_lterm_liab']]
