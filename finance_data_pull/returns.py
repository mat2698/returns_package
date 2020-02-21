# import some modules
import os
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import collections


def create_data(ticker, start_date, end_date, frequency=None):
    old_date_string = start_date
    decrement = 2
    year = int(old_date_string[:-3])
    month = int(old_date_string[-2:])
    month = month - decrement
    if month < 1:
        month = 12
        year = year - 1
    new_start_date = str(year) + '-{:02d}'.format(month)

    # Creates vector containing all dates in period
    dates = pd.date_range(new_start_date, end_date)
    dates = dates.to_frame()

    # Pulls historical price data for given ticker
    item_data = web.DataReader(ticker, 'yahoo', new_start_date, end_date)

    # Merges price data with full set of dates to account for missing dates.
    updated_data = dates.merge(right=item_data['Adj Close'], how='left', left_index=True, right_index=True)
    updated_data = updated_data.fillna(method='ffill')

    if (frequency == 'monthly'):
        month_ends = pd.date_range(new_start_date, end_date, freq='M')
        month_ends = month_ends.to_frame()
        updated_data = month_ends.merge(updated_data['Adj Close'], left_index=True, right_index=True)

    return (updated_data)


def create_dataset(benchmark, portfolio, date_range):
  start_date = date_range[0]
  end_date = date_range[1]
  bench_data = create_data(benchmark, start_date, end_date, 'monthly')['Adj Close']
  portfolio_data = create_data(portfolio, start_date, end_date, 'monthly')['Adj Close']
  t_bill = create_data('^IRX', start_date, end_date, 'monthly')['Adj Close']

  dataset = pd.DataFrame({'13_week_close': t_bill, 'bench_close': bench_data, 'portfolio_close': portfolio_data})

  dataset['bench_return'] = dataset['bench_close'].pct_change()
  dataset['portfolio_return'] = dataset['portfolio_close'].pct_change()
  dataset['t_bill_return'] = dataset['13_week_close'].pct_change()

  dataset['bench_excess_returns'] = dataset['bench_return'] - dataset['t_bill_return']
  dataset['portfolio_excess_returns'] = dataset['portfolio_return'] - dataset['t_bill_return']

  dataset['bench_downside_returns'] = np.nan
  dataset['portfolio_downside_returns'] = np.nan

  dataset.loc[dataset['bench_return'] < 0, 'bench_downside_returns'] = dataset['bench_return']
  dataset.loc[dataset['portfolio_return'] < 0, 'portfolio_downside_returns'] = dataset['portfolio_return']

  return(dataset.drop(dataset.index[0:1]))


import numpy as np


def calc_compound_ann(r):
    ret = (((1 + r.mean()) ** 12) - 1) * 100
    return (ret)


def calc_simple_ann(r):
    ret = r.mean() * 12 * 100
    return (ret)


def format_decimal(x):
    res = format(x, '.2f')
    return (res)


def max_drawdown(returns):
    maximums = np.maximum.accumulate(returns)
    drawdowns = 1 - returns / maximums
    return np.max(drawdowns)


def max_dur_drawdown(returns):
    maximums = np.maximum.accumulate(returns)
    drawdowns = 1 - returns / maximums
    count = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] < 0:
            start = i
            for val in drawdowns[i:]:
                count += 1
                if val >= 1:
                    end = i + count
                    break
            break

    return (end - start)


from pprint import pprint
import numpy as np


def calc_maxdd(ret):
    # calculate max drawdown and max drawdown duration
    results = _calc_maxdd(ret)  # internal routine returns triplets (tend, nper, val)
    return results


def _calc_maxdd(ret):
    ddp = _calc_ddp(ret)
    maxdd_idx = np.nan
    maxdur_idx = np.nan
    maxdd = np.nan
    maxdur = np.nan
    # print(type(ddp))
    if len(ddp) == 0:
        return {}
    for idx, d in enumerate(ddp):
        if idx == 0:
            maxdd_idx = idx
            maxdd = d['maxdd']
            maxdur_idx = idx
            maxdur = d['nper']
        else:
            if d['maxdd'] < maxdd:
                maxdd = d['maxdd']
                maxdd_idx = idx
            if d['nper'] > maxdur:
                maxdur = d['nper']
                maxdur_idx = idx
    results = {
        'maxdd': ddp[maxdd_idx]
        , 'maxdur': ddp[maxdur_idx]
        , 'ddp': ddp
    }
    return results


def _calc_ddp(ret):
    # check for na; return immediately if any na's
    if np.any(np.isnan(ret)):
        return []
    # calculate cumulative value series from return series
    # and prepend a value of 1.0 for the cumulative value series
    val = np.cumprod(1 + np.array(ret))
    val = np.insert(val, 0, 1.0)
    # identify all drawdown periods (not sub-drawdown periods)
    ddp = list()  # initialize array of drawdown periods
    cnt = 0
    i_end = 0
    while i_end < len(val) - 1 and cnt <= 10e10:  # counter just in case
        i_beg = i_end
        val_beg = val[i_beg]
        # check if we are starting a new drawdown
        if val[i_beg + 1] / val[i_beg] >= 1.0:  # we are not starting a drawdown period
            i_end = i_beg + 1  # move forward one period
        else:  # we are starting a drawdown period
            max_drawdown = np.inf
            for i in range(i_beg + 1, len(val)):
                i_end = i
                max_drawdown = min(max_drawdown, val[i] / val_beg)
                if val[i] / val_beg > 1.0 or i_end >= len(val) - 1:  # recovery! so found end of drawdown period
                    nper = i_end - i_beg
                    # return index is one less; need to adjust i_end
                    d = {'i_beg': i_beg, 'i_end': i_end - 1, 'nper': i_end - i_beg, 'maxdd': max_drawdown}
                    ddp.append(d)
                    break
    return ddp


def get_stats(dataset, benchmark, portfolio, start, end):
    result = collections.OrderedDict()
    result['bmk_id'] = benchmark
    result['p_id'] = portfolio
    result['start'] = start
    result['end'] = end

    # Calculate average excess return
    result['avg_ret'] = np.round(calc_simple_ann(dataset['portfolio_return']), 2)

    # Calculate standard deviation
    result['std'] = np.round(dataset['portfolio_return'].std() * np.sqrt(12) * 100, 2)

    # Calculate sharpe ratio
    result['sharpe'] = np.round(result['avg_ret'] / result['std'], 2)

    # Calculate sortino ratio
    result['sortino'] = np.round(result['avg_ret'] / (dataset['portfolio_downside_returns'].std() * np.sqrt(12) * 100),
                                 2)

    # Calculate M2
    result['mm'] = np.round((result['sharpe'] * result['std']) + dataset['t_bill_return'].mean(), 2)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OLS Regression Values
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Set independent variable.
    x1 = dataset['bench_return'][2:]
    # Add a constant to be included in the regression.
    X = sm.add_constant(x1)

    # Run regression
    results_portfolio = sm.OLS(dataset['portfolio_return'][2:], X).fit()

    # OLS Results for SP500
    result['alpha'] = np.round((results_portfolio.params['const'] * 100 * 12), 2)
    result['beta'] = np.round(results_portfolio.params['bench_return'], 2)
    result['treynor'] = np.round(result['avg_ret'] / result['beta'], 2)
    result['appraisal_ratio'] = np.round(result['alpha'] / (
                np.sqrt((1 - results_portfolio.rsquared) * np.var(dataset['portfolio_return'], ddof=0) * 12) * 100), 2)
    result['r-sq'] = np.round(results_portfolio.rsquared, 2)
    result['resid-std'] = np.round(
        (np.sqrt((1 - results_portfolio.rsquared) * np.var(dataset['portfolio_return'], ddof=0)) * 12), 2)

    # Max drowdown
    res = calc_maxdd(dataset['portfolio_return'].tolist())

    if "maxdd" in res:
        result['maxdd'] = np.round((res['maxdd']['maxdd'] - 1) * 100, 2)
    else:
        result['maxdd'] = np.nan

    # Maximum duration drawdown (max time to recovery)
    if "maxdur" in res:
        result['maxdur'] = np.round(res['maxdur']['nper'] - 1, 2)
    else:
        result['maxdur'] = np.nan

    # Calculate Mo VaR95 (v)
    result['Mthly_VaR95_v'] = np.round(dataset['portfolio_return'].mean() - (1.65 * dataset['portfolio_return'].std()),
                                       2)

    # Calculate Mo VaR95 (v)
    result['Mthly_CVaR95_v'] = np.round(
        (dataset['portfolio_return'].mean() - (2.32 * dataset['portfolio_return'].std())), 2)

    # Calculate Mo VaR95 (h)
    result['Mthly_VaR95_h'] = np.round(np.quantile(dataset['portfolio_return'], (1 - 0.95)), 2)

    # Calculate Mo VaR95 (h)
    result['Mthly_CVaR95_h'] = np.round(
        dataset['portfolio_return'].mean() - (2.063 * dataset['portfolio_return'].std()), 2)

    return (result)


def calc_results(benchmark, portfolios, date_range):
    start_dates = [row[0] for row in date_range]
    end_dates = [row[1] for row in date_range]

    stats = []
    headers = ['']

    for start, end in zip(start_dates, end_dates):
        analysis = collections.OrderedDict()
        analysis['daterange'] = [start, end]

        # Add benchmark results
        data = create_dataset(benchmark, benchmark, [start, end])
        bench_data = get_stats(data, benchmark, benchmark, start, end)
        analysis['bmk'] = bench_data

        portfolio_stats = []
        headers.append(benchmark + "from " + start + "to " + end)
        for portfolio in portfolios:
            headers.append(portfolio + "from " + start + "to " + end)
            data = create_dataset(benchmark, portfolio, [start, end])
            portfolio_data = get_stats(data, benchmark, portfolio, start, end)

            portfolio_stats.append(portfolio_data)

        analysis['all_portfolios'] = portfolio_stats
        stats.append(analysis)

    return (stats, headers)


# Creates a table to show data based upon an analytic object.
def generate_table(stats, headers):
    statistics = []
    statistics.append(list(stats[1]['bmk'].keys()))
    for i in range(len(stats)):
        statistics.append(list(stats[i]['bmk'].values()))
        for j in range(len(stats[1]['all_portfolios'])):
            statistics.append(list(stats[i]['all_portfolios'][j].values()))

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, align="center", fill=dict(color='#C2D4FF'), font=dict(color='white', size=14)),
        cells=dict(values=statistics, fill=dict(color='#f2f3f4'), font=dict(size=14), height=30))])

    fig.update_layout(
        autosize=False,
        width=1500,
        height=900)

    fig.show()


if __name__ == "__main__":
    print(" ")