# %% [markdown]
# # IE 582 - HW 1 - Stock Exchange Data
# 
# Prepared by Abdullah Kayacan

# %% [markdown]
# ## Introduction

# %% [markdown]
# ### Module and Data Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# %%
data_raw = pd.read_csv('data/all_ticks_wide.csv.gz')
trends_data_raw = pd.read_csv("data/multiTimeline.csv",skiprows=1)

# %% [markdown]
# ## Descriptive Analytics

# %% [markdown]
# In this section, we start with exploring the basic properties of the data.

# %%

tickers = data_raw.columns[1:].values
print(f"Stocks available in data:\n{tickers}")

# %%
data_raw.head()

# %% [markdown]
# ### Domain-Specific Anomalies

# %% [markdown]
# Before diving deep into the data, determining and fixing the domain-specific abnormalities might be helpful. We know that daily min and max price change is no more than %20 before 2020 [1]. Thus, any price change more than ${1.2}^2-1$ or less than ${0.8}^2-1$ is not possible.
# 
# [1] https://www.borsagundem.com.tr/haber/borsada-tavan-marji-yeniden-yuzde-20ye-ciksin-1479622#:~:text=Borsa%20İstanbul%27da%20aşırı%20fiyat,devre%20kesici%20uygulamasında%20değişikliğe%20gidildi.

# %% [markdown]
# The code below gives us the indices of extreme changes:

# %%
changes = data_raw[tickers].iloc[1:].values/data_raw[tickers].iloc[:-1]
problematic_instances = np.where((changes > 1.2*1.2) | (changes < 0.8*0.8))
problematic_instances = np.array([problematic_instances[0]+1,problematic_instances[1]]).transpose()
del changes
problematic_instances

# %% [markdown]
# One of the most repeating abnormalitiy is on the 4159th row. Let us check the summary of this row.

# %%
data_raw[tickers].iloc[4159].describe()

# %% [markdown]
# The maximum price data available in that row is 0.0001 TRY which is not possible in BIST. Thus we will nullify the row. Let us check the remaining anomalies:

# %%
problematic_instances = problematic_instances[~np.isin(problematic_instances[:,0],[4159,4160])]
problematic_instances

# %% [markdown]
# We have very few anomalities on the remaining rows. Let us see the changes on the rows. The table below show the abnormal values and their neighbors in the data. The column names indicates the stock and its relevant row with abnormality.

# %%
problematic_stocks = pd.DataFrame()

problematic_instances
for problematic_instance in problematic_instances:
    problematic_tick,ticker_index = problematic_instance
    problematic_stock = data_raw.iloc[problematic_tick-2:problematic_tick+5,[ticker_index+1]].reset_index(drop=True)
    problematic_stock.columns = [problematic_stock.columns[0]+f"_{problematic_tick}"]
    problematic_stocks = pd.concat([problematic_stocks,problematic_stock],1)

print(problematic_stocks)

# %%
print(data_raw.iloc[390]["timestamp"])

# %% [markdown]
# BAGFS seems to have a stock split (200%) on 2012-10-08. We checked and found the relevant dislosure in [2]. The other price abnormalities seem to be data errors since the price levels are maintained after one period. We make the necessary corrections with the code snippet below.
# 
# [2] https://www.kap.org.tr/tr/Bildirim/239817

# %%
data_corrected = data_raw.copy()
data_corrected.loc[390:,"BAGFS"] = data_corrected.loc[390:,"BAGFS"]*3
data_corrected.loc[7938,"GUBRF"] = np.nan
data_corrected.loc[18651,"GOODY"] = np.nan
data_corrected.loc[18934,"GOODY"] = np.nan
data_corrected.loc[4159,tickers] = np.nan#stocks_corrected.drop(4159,0)
data_corrected = data_corrected.reset_index(drop=True)
data_corrected["year_month"] = [str(e)[:7] for e in data_corrected["timestamp"]]

# %% [markdown]
# It might be useful to add some period columns which indicates the month and quarter of the specific time. We will use them later 

# %%
data_corrected["year_month"] = [str(e)[:7] for e in data_corrected["timestamp"]]
data_corrected["year_quarter"] = [e[:5]+str((int(e[-2:])-1)//3+1) for e in data_corrected["year_month"].values]

data_corrected[["year_month","year_quarter"]].drop_duplicates().head()

# %% [markdown]
# ### Data Structure

# %% [markdown]
# It is convenient to examine the data structure by visualization. Histogram and line plots are the essential tools to explore the data.

# %% [markdown]
# The chart below shows the price histograms of all the stocks available.
# - As can be seen on the plot, most of the prices are roughly normally or log-normally distributed. Right-skewed distributions make sense since the price movement steps get wider as the prices increase.
# - Bimodal distributions might indicate that at some point of the time, the stock gained a significant value and maintained it. It might stem from a huge capital increase or new deals of the company.
#     - Another reason for bimodal distribution might be the decrease in prices due to stock splits or excessive dividents distributed. One should check it case by case to be sure.

# %%
fig,axs = plt.subplots(nrows=10,ncols=6,figsize=(12,20))
plt.suptitle("Price Histograms")
plt.tight_layout(rect=(0,0,1,.98),h_pad=2)
for i in range(len(tickers)):
    ticker = tickers[i]
    axs[i//6][i%6].set_title(ticker)
    axs[i//6][i%6].hist(data_corrected[ticker])

# %% [markdown]
# Since our data is heavily time-dependent, we should also check the line plots. Especially in non-stationary data, histograms might be pretty misleading.
# 
# 15-min data is hard to visualize in a line plot. Thus, we will observe the monthly progress of average prices to see their trends and seasonalities.
# 
# We see from the plot below that:
# - Most of the stocks have an upward trend. It is reasonable since BIST100 index is also increased significantly in this time period [3].
# - Some stocks show periodic patterns (eg. SAHOL, YKBNK) which might have several reasons such as periodic divident distribution.
# - Some industries have common patterns.
#     - Both SISE and TRKCM (glass products manufacturers) have prominent and stady price increase.
#     - AKBNK, ALBRK, GARAN, and YKBNK (banks) have a similar cyclic pattern. 

# %%
data_monthly = data_corrected.groupby("year_month")[tickers].mean().reset_index()

fig,axs = plt.subplots(nrows=10,ncols=6,figsize=(12,20))
plt.suptitle("Price Timelines")
plt.tight_layout(rect=(0,0,1,.98),h_pad=4)
for i in range(len(tickers)):
    ticker = tickers[i]
    axs[i//6][i%6].set_title(ticker)
    axs[i//6][i%6].plot(data_monthly["year_month"],data_monthly[ticker])

    axs[i//6][i%6].xaxis.set_major_locator(plt.MaxNLocator(5))
    [e.set_rotation(35) for e in axs[i//6][i%6].get_xticklabels()]

# %% [markdown]
# Another useful visualization idea might be the histogram of price changes. While timeline plots give some insights for long term traders, price change histograms might be more suitable for the ones who want to see the summary of short term (15 min) price moves.
# 
# - As expected, most of the price changes in 15 minutes are very close to zero. Thus, we use a log scale on y-axis of subplots to show frequencies.
# - Monst of the price chanes distributed normally with leptocurtosis and very low skewness.

# %%
changes = data_corrected.copy()
changes[tickers] = changes[tickers].iloc[1:]/changes[tickers].iloc[:-1].values-1

fig,axs = plt.subplots(nrows=10,ncols=6,figsize=(12,20))
plt.tight_layout(rect=(0,0,1,.95),h_pad=2)
plt.suptitle("Histograms of Returns\n(y-axis is log-scaled)")
for i in range(len(tickers)):
    ticker = tickers[i]
    axs[i//6][i%6].set_title(ticker)
    axs[i//6][i%6].hist(changes[ticker],bins=15)
    axs[i//6][i%6].set_yscale("log")


# %% [markdown]
# Since price changes are kind of normalized version of prices, we can plot them together in one histogram. As expected, it has a similar distribution with the single-stock histograms.

# %%
plt.title("Histogram of Returns\n(y-axis is log-scaled)")
plt.hist(changes[tickers].values.flatten(),bins=15)
plt.yscale("log")

# %% [markdown]
# ### Summary Statistics

# %% [markdown]
# Tendency, dispersion, and shape characteristics of the stocks are summarized in the table below.
# - At a glance, stocks seems in a reasonable range although it is hard to comprehend since price levels differ among the tickers.
# - Most of the prices seem slightly platykurtic, which means that the mode of the distribution is wide. One of the reason of it for some stoks might be the multi modal distribution of the prices.
# - Most of the prices are positive skewed as expected due to the higher price moves in higher prices.

# %%
pd.set_option('display.max_columns', 70)
price_summary = data_corrected.describe()
price_summary.loc['skewness'] = data_corrected.skew().values
price_summary.loc['kurtosis'] = data_corrected.kurtosis().values
price_summary

# %% [markdown]
# In order to see more details on the summary statistics, we prepared boxplots. Since it would be hard to draw all boxes into one plot, we ordered the stocks by price and grouped accordingly. We plot prices of 10 stocks in similar price level side by side in each chart.
# 
# - Skewness of the prices are eminent as in sumarry statistics table.
# - We observe outliers in some stocks according to boxplot outlier rule (1.5 IQR). They are more severe in stocks with lower price levels.
# 

# %%
tickers_ordered_by_mean = price_summary.transpose()["mean"].sort_values().index.values
long_format = data_corrected[tickers].melt(var_name="ticker",value_name="price").dropna()
for i in range(6):
    tickers_used = tickers_ordered_by_mean[i*10:(i+1)*10]
    data_for_boxplot = long_format[np.isin(long_format["ticker"],tickers_used)]
    plt.figure(figsize=(12,5))
    plt.title(f"Boxplot of Prices - Price Level Group {i+1}")
    sns.boxplot(data_for_boxplot,x="ticker",y="price")
    plt.show()

# %% [markdown]
# Applying same summarization techiques on price changes can also be useful. The summary table and boxplots can be found below.
# - Mean and median values are either zero or very close to zero. It is expected since the period of change we looked for is very narrow (15 mins).
# - For the similar reason, standard devation for each of them is also very low.
# - As we have already observed in the previous histograms, we see very high kurtosis values.
# - Positive skewness of the data is more visible in the summary statistics compared to the histograms we plotted before.

# %%
change_summary = changes.describe()
change_summary.loc['skewness'] = changes.skew().values
change_summary.loc['kurtosis'] = changes.kurtosis().values
change_summary

# %% [markdown]
# For the change data boxplots have too much outliers due to the low inter-quartile ranges. Because of that it is hard to interpret the boxplots for this one.

# %%
long_format = changes[tickers].melt(var_name="ticker",value_name="price").dropna()
for i in range(6):
    tickers_used = tickers_ordered_by_mean[i*10:(i+1)*10]
    data_for_boxplot = long_format[np.isin(long_format["ticker"],tickers_used)]
    plt.figure(figsize=(12,5))
    plt.title(f"Boxplot of Prices - Price Level Group {i+1}")
    sns.boxplot(data_for_boxplot,x="ticker",y="price")
    plt.show()

# %% [markdown]
# ### Null Value Analysis

# %% [markdown]
# Null values are worth to investigate since they might effect the quality of further analyses. Knowing how much data will disrupt the analysis is important.

# %% [markdown]
# The graph below shows the share of null rows for each column (or stock).
# - ISDMR has null rows above 70% which might be very disruptive for the rest of the tickers if ignored.
# - After ISDMR, rate of null rows gradually decreases for column starting from about 15%.
# - Most of the columns have approximately 1.5% null values.

# %%
pd.DataFrame((data_corrected.isna().sum()/data_corrected.shape[0]).sort_values(ascending=False)) \
    .plot(figsize=(15,6),kind="bar",legend=False,title="Share of Null Values")
plt.show()

# %% [markdown]
# We plotted below the histogram of the number of null values per row.
# - Most of the rows include few null values.
# - There are a significant number of rows which have null values between 55 and 60.

# %%
plt.title("Histogram of # of Nulls per Row")
data_corrected.set_index("timestamp").isna().sum(1).hist(bins=60,figsize=(15,6))
plt.show()

# %% [markdown]
# The plot below shows the monthly average number of nulls per 15-min ticks.
# - There is a pick at the first months of 2013.
# - Number of null rows slightly decreases with time.

# %%
null_history = data_corrected.set_index("timestamp").isna().sum(1).reset_index().rename(columns={0:"null"})
null_history["year_month"] = [str(e)[0:7] for e in null_history["timestamp"].values]
null_history = null_history.groupby("year_month")["null"].mean()
plt.figure(figsize=(15,6))
plt.plot(null_history)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.show()

# %% [markdown]
# ## Correlations & Time Window Correlations

# %% [markdown]
# ### Correlations

# %% [markdown]
# Let us investigate the general correlation among the stock prices. Since correlation done in a pairwise manner and null values are dropped by default, we will not apply any treatment for null values.
# 
# The plot below shows the heatmap of price correlations and the hierarchical clusters according to the correlations.
# 
# - Most of the correlation values seem to be positive. It might be anticipated since there is a probable positive correlation between macroeconomic indices (eg. inflation rate, BIST100 index) and each stock.

# %%
price_corr = data_corrected.corr()
plt.figure(figsize=(40,40))
sns.clustermap(price_corr,xticklabels=True,yticklabels=True)
plt.show()

# %% [markdown]
# Let us see the most correlated stock prices.
# 
# - As shown in the output below, most of the positive correlations have similar values. This might stem from multicollinearity due to the macroeconomic indicators or stock exchange index. Since we do not have any stock exchange index data, we can only assume it.
# - We cannot make any interpretation by just looking the correlated stocks. The common point of positive correlated stocks might be that nearly all the stocks are from manufacturing industry.

# %%
stocks_corr_melted = price_corr.stack().reset_index()
stocks_corr_melted.columns = ["stock1","stock2","corr"]
stocks_corr_melted = stocks_corr_melted[stocks_corr_melted["stock1"] < stocks_corr_melted["stock2"]]
stocks_corr_melted = stocks_corr_melted.sort_values("corr",ascending=True)

print("Most Positive Correlations:")
print(stocks_corr_melted[::-1][:5].head())
print("\n-----\n")
print("Most Negative Correlations:")
print(stocks_corr_melted[:5].head())

# %% [markdown]
# ### Time Window Correlations

# %% [markdown]
# Time window correlation might be helpful to examine the stability of correlations. We choosen two pairs of stocks:
# - AKBNK-GARAN: Seemed to have common cycles as we investigated in the previous section.
# - SISE-TRKCM: Both have similar upward trend in monthly summarized line plots.
# 
# As shown in the plots below. Both of them keeps positive their correlations throughout the months and quarters.

# %%
stock_pairs = np.array([["AKBNK","GARAN"],["SISE","TRKCM"]])


for period_type,title_period in np.array([["year_month","Month"],["year_quarter","Quarter"]]):
    fig,axs = plt.subplots(1,2,figsize=(20,5))
    plt.suptitle(f"Time Window Correlations (Window: {title_period})")
    plt.tight_layout()

    for i,(stock_1,stock_2) in enumerate(stock_pairs):
        
        axs[i].set_title(f"{stock_1}-{stock_2}")

        corr_pair_data = data_corrected[["timestamp",period_type,stock_1,stock_2]].dropna()
        corr_pair = corr_pair_data.groupby(period_type).corr().reset_index().drop([stock_2,"level_1"],1).groupby(period_type).min().rename(columns={stock_1:"corr"})
        axs[i].plot(corr_pair.index,corr_pair["corr"])
        [e.set_rotation(45) for e in axs[i].get_xticklabels()]
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()


# %% [markdown]
# ## PCA

# %% [markdown]
# PCA can only be conducted on the rows without null values.

# %%
f"Rate of rows with null values: {(data_corrected.isna().sum(1) > 0).mean():.2%}"

# %% [markdown]
# If we conduct the PCA on this data, only 18.5% of the data will be used. This will probably lead to incorrect inferences. We can use three additional methods before PCA to mitigate null value problem:
# - Aggregate prices by combining some periods (i.e. using hourly or daily data instead of 15-minutes)
# - Discard a few features with the most null values.
# - Fill nulls with an imputation method.

# %%
pca_data_0 = data_corrected[tickers].dropna()

# %% [markdown]
# #### Null Treatment I - Aggregation

# %%
data_corrected["hour"] = [e[:16] for e in data_corrected["timestamp"].values]
data_corrected["day"] = [e[:10] for e in data_corrected["timestamp"].values]

data_hourly = data_corrected.groupby("hour")[tickers].mean().reset_index()
print(f"Rate of rows with null values (hourly data): {(data_hourly.isna().sum(1) > 0).mean():.2%}")

data_daily = data_corrected.groupby("day")[tickers].mean().reset_index()
print(f"Rate of rows with null values (daily data): {(data_daily.isna().sum(1) > 0).mean():.2%}")

# %% [markdown]
# As can be seen in the output of the snippet above:
# - Hourly aggregated data roughly the have same null rate.
# - More than half of the daily aggregated data is null. Plus, we lose the intra-day price dynamics.
# - Still, we will give daily aggregation a try.

# %%
pca_data_1 = data_daily[tickers].dropna()
del data_hourly,data_daily

# %% [markdown]
# #### Null Treatment II - Discrading Highly Null Features

# %% [markdown]
# The output below shows the null rates after removing the stocks one by one starting from the most null values.

# %%
tickers_ordered_by_null_share = pd.DataFrame((data_corrected.isna().sum()/data_corrected.shape[0]).sort_values(ascending=False)).index.values
tickers_ordered_by_null_share

# %%
for i in range(11):
    print(f"Rate of rows with null values (After discarding {i} stock(s)): {(data_corrected.loc[:,tickers_ordered_by_null_share[i:]].isna().sum(1) > 0).mean():.2%}")

# %% [markdown]
# - We will discard 5 stocks from the data. Although not using the available features is not usual, the disruptive effect of those stocks would be eliminated by this method.

# %%
pca_data_2 = data_corrected.loc[:,tickers_ordered_by_null_share[5:]].dropna()

# %% [markdown]
# #### Null Treatment III - Imputation

# %% [markdown]
# The imputation method allow us to use all the instances and features available in data. However, it might make up spurious correlations especially when null values coexist in same rows for the feautres in the original data. Thus, it might deteriorate the quality of PCA, which is heavily dependent on correlations.
# 
# - Below, we have applied linear interpolation. After than that, we used forward fill and backward fill functions to fill the nulls at the begining and the end of the table.

# %%
pca_data_3 = data_corrected[tickers].copy()
pca_data_3 = pca_data_3.interpolate().ffill().bfill()

# %% [markdown]
# #### PCA Implementation

# %% [markdown]
# - We applied PCA on data with imputation (pca_data_3) since other methods have too much instance or feature loss.
# - We used standard scaler in our pipeline to prevent the unnecessary variance reduction in stock prices with wide price ranges.
# - You can see the cumulative variance ratio of each component below.
# - Cumulative variance ratio starts with 50% which is not bad. First 6 components are enough the capture the 90% of the variablity of the data.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA())])

pca_output = pipeline.fit_transform(pca_data_3)
pipeline["pca"].explained_variance_ratio_.cumsum()

# %% [markdown]
# - We can examine the loadings of the first 6 components with the compononets_ attribute below.

# %%
first_six_components = pipeline["pca"].components_[:6]
first_six_components

# %% [markdown]
# We reported the stocks which have the most positive and most negative 3 loadings in each components for the first six PC.
# - As we expected, prominent positive loadings of the first component were highly correalted in the original data.
# - On the other, although being negative, loadings of hand banking companies are also high absolute value.

# %%
pca_columns = pca_data_3.columns.values
most_prominent_factors = first_six_components.argsort()[:,[-1,-2,-3,2,1,0]]
most_prominent_factors = pd.DataFrame(pca_columns[most_prominent_factors])
most_prominent_factors.columns = ["first_positive_loading","second_positive_loading","third_positive_loading",
                                  "third_negative_loading","first_negative_loading","second_negative_loading"]
most_prominent_factors

# %% [markdown]
# Below we see the visualization of data according to first two PCs.
# - Two components explains 68% of the variablitiy, which not much reliable to make further analysis.
# - The components have very different patterns as shown below. This might be an indication of certain price dynamics in different time periods. Clustreing might help us for exploring the patterns in those period of times.  

# %%
plt.scatter(pca_output[:,0],pca_output[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Summary of Prices in Two PCs")
plt.show()

# %% [markdown]
# Let us see the timeline of each component together.
# - As shown in the plot below, PCs seem uncorrelated as expected.
# - PC1 seems like the overall trend in the data, probably it would be very correlated with BIST100 index.
# - PC2 seems like to represent the short term periodic cycles in the data.
# - PC3 seems like a long term periocidity in the data.
# - One can see the similarity between stock movements we have seen in line plots in previous section and the PC lines below.

# %%
plt.figure(figsize = (12,5))
for i in range(6):
    plt.plot(pca_output[:,i])
plt.legend([f"PC{i+1}" for i in range(6)])
plt.xlabel("Time indices (15-mins)")
plt.ylabel("Components")
plt.title("Timeline of the Initial PCs")
plt.show()

# %% [markdown]
# ## Google Trends

# %% [markdown]
# We want to check the news related with AKBNK, GARAN, SISE, and TRKCM (the stocks that we investigate in correlation analysis).
# - Search URL: https://trends.google.com/trends/explore?date=2012-09-17%202019-07-23&geo=TR&gprop=news&q=%2Fg%2F12fzdqdn9,%2Fg%2F12fzdqc6p,%2Fg%2F12fzdq9mr,%2Fm%2F0cp7qbj
# - You can see a few rows in the table below.
# - We selected news trends instead of web search trends.
# - For the date-range we choose, only monthly data is available. Thus, we conduct our analysis with monthly price data.

# %%
trends_data_raw.head()

# %%
trends_data = trends_data_raw.copy()
trends_data.columns = ["year_month","AKBNK_news","GARAN_news","SISE_news","TRKCM_news"]

# %%
trends_data.set_index("year_month").plot(title="Google News Trends",figsize=(15,5))
plt.show()

# %% [markdown]
# We checked the correlations between stoks and their related news.
# - Interestingly, there seems no significant correlation between news and prices.
# - The reason might be the high number of zero values in news data.
# - Thus, looking point by point for the news events might be better.

# %%
price_news_combined = pd.merge(
    data_monthly[["year_month","AKBNK","GARAN","SISE","TRKCM"]],
    trends_data,
    "outer"
)

news_correlations = price_news_combined.corr().stack().reset_index()
news_correlations.columns = ["ticker","news","corr"]
important_pairs = pd.DataFrame(np.array([
    ["AKBNK","AKBNK_news"],
    ["AKBNK","GARAN_news"],
    ["GARAN","AKBNK_news"],
    ["GARAN","GARAN_news"],
    ["SISE","SISE_news"],
    ["SISE","TRKCM_news"],
    ["TRKCM","SISE_news"],
    ["TRKCM","TRKCM_news"],
]),columns = ["ticker","news"])

important_pairs.merge(news_correlations,"left")

# %% [markdown]
# Plots of news and prices are shown below.
# - There seems like a negative relationship between banking news and stock prices. The news might include some negative contents.
# - There is no apparent relation between glass industry stocks and their news trends.
# - Monthly data might be too wide to see the effect of the news. One can examine the news one by one and observe the price changes around the arrival time to see a more detailed analysis.

# %%
for ticker,news in important_pairs.values:
    news = news.split("_")[0]
    fig, ax1 = plt.subplots(figsize=(15,5))
    ax2 = ax1.twinx()

    price_news_combined.set_index("year_month")[[ticker]].plot(ax=ax1,legend=False,color="orange")
    price_news_combined.set_index("year_month")[[news+"_news"]].plot(ax=ax2,color="gray",legend=False)
    ax1.legend([ticker],loc=2)
    ax2.legend([news+" News"],loc=1)
    ax1.set_ylabel("Price (₺)")
    ax2.set_ylabel("News density")
    plt.title("Price vs. News")
    plt.show()


