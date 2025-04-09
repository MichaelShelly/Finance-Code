#%%
import yfinance as yf
import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from scipy.stats import linregress,skew, kurtosis, norm
from scipy.optimize import minimize, norm, skew, kurtosis
from scipy.stats.mstats import gmean
#%%
# Stores inputs
investment_data = []
transaction_type = []
transaction_date = []

while True:
    trading_decisions = input("\nInput 'buy' or 'sell' to indicate if your trade was a buy or sell, or input 'cash' to represent cash \nbeing added or taken from the portfolio (or type 'done' to finish): ").strip()
    
    if trading_decisions not in ["buy", "sell", "cash", "done"]:
        print("Invalid input. Please enter 'buy', 'sell', or 'cash'.")
        continue

    if trading_decisions == "done":
        break
    
    elif trading_decisions == "buy":
        investment_decision = input("Which stock did you buy?").upper()
        investment_date = input("What day did you buy it? (YYYY-MM-DD)")
        investment_amount = float(input("How many shares did you purchase?"))
        
    elif trading_decisions == "sell":
        investment_decision = input("Which stock did you sell?").upper()
        investment_date = input("What day did you sell it? (YYYY-MM-DD)")
        investment_amount = float(input("How many shares did you sell?"))
        
    elif trading_decisions == "cash":
        investment_amount = float(input("How much money did you deposit/withdrawl to your account?"))
        investment_date = input("When did you add it to your account? (YYYY-MM-DD)")
        transaction_type = input("Was this a deposit or withdrawal? (d for deposit, w for withdrawal): ")
        
        if transaction_type.lower() == 'd':
            investment_decision = "Deposit"
        elif transaction_type.lower() == 'w':
            investment_decision = "Withdrawal"
        else:
            print("Invalid input for transaction type.")
            continue
    
    # Appending data to investment_data list
    investment_data.append({
        "Investment Action": trading_decisions,
        "Stock": investment_decision, 
        "Transaction Date": investment_date, 
        "Number of Shares": investment_amount
    })
#%%
# removes temporary variables
del trading_decisions, transaction_type, investment_date, investment_decision, investment_amount

investment_data = pd.DataFrame(investment_data)

#%% 
# Pulling data from yfinance
## Filters data we need
filtered_data = investment_data[~investment_data['Stock'].str.contains('Deposit|Withdrawal', na=False)]
if not filtered_data.empty:
    filtered_data = filtered_data.sort_index()

#%%
stocks_to_pull = filtered_data.groupby('Stock').head(1)

stocks_to_pull = stocks_to_pull[['Stock', 'Transaction Date']].reset_index(drop=True)

#pulls from yfinance

today = datetime.today().date()
stock_data = []


for index, row in stocks_to_pull.iterrows():
    stock = row['Stock']
    transaction_date = row['Transaction Date']
    

    stock_info = yf.download(stock, start=transaction_date, end=today)
    

    globals()[stock] = stock_info[['Open', 'Close']]


#%%
#makes dict of stock data
dict_of_stocks = {stock: globals()[stock] for stock in stocks_to_pull['Stock']}

#removes multi-index
for ticker, df in dict_of_stocks.items():
    df.columns = [col[1] for col in df.columns]  # Flatten to just column names like "Open", "Close"

#forces columns to be called "open" and "close"
for ticker, df in dict_of_stocks.items():
    df.columns = ['Open', 'Close']
#%% Handling Non-Trading Days
investment_data["Transaction Date"] = pd.to_datetime(investment_data["Transaction Date"])  # Convert to datetime

for i, row in investment_data.iterrows():
    stock = row["Stock"]
    date = row["Transaction Date"]
    
    if stock in dict_of_stocks:
        stock_data = dict_of_stocks[stock]
        valid_dates = stock_data.index  # Get all valid trading dates
        if date not in valid_dates:
            date = valid_dates[valid_dates >= date].min()  # Find the next available trading day
        investment_data.at[i, "Transaction Date"] = date

#%% Track total invested capital
portfolio = {'cash': 0}  # Initialize portfolio with cash
total_invested = 0
total_shares_bought = {}

if not filtered_data.empty:
    for _, row in investment_data.iterrows():
        action = row["Investment Action"].lower()
        stock = row["Stock"]
        date = row["Transaction Date"]
        amount = row["Number of Shares"]

        if stock == "cash":
            if action == "deposit":
                portfolio['cash'] += amount
                total_invested += amount
            elif action == "withdrawal":  # Fixed indentation here
                portfolio['cash'] -= amount
                total_invested -= amount
            continue  # Continue should apply to the entire "cash" case

        stock_data = dict_of_stocks[stock]
        if date not in stock_data.index:
            continue

        stock_price = stock_data.loc[date, "Open"] if action == "buy" else stock_data.loc[date, "Close"]
        shares = amount

        portfolio[stock] = portfolio.get(stock, 0)  # Ensure stock exists in portfolio
        total_shares_bought[stock] = total_shares_bought.get(stock, 0)  # Ensure stock exists in tracking

        if action == "buy":
            portfolio[stock] += shares
            total_shares_bought[stock] += shares
            total_invested += amount
        elif action == "sell":
            portfolio[stock] -= shares

if not investment_data.empty:
    portfolio['cash'] = investment_data.iloc[0]["Number of Shares"]  
#%%
# Compute geometric return
#Sets dates
start_date = pd.Timestamp(investment_data["Transaction Date"].min())
end_date_input = input("Enter end date (YYYY-MM-DD) or 'today': ")
if end_date_input.lower() == "today":
    end_date = pd.Timestamp(datetime.today().date())
else:
    end_date = pd.Timestamp(end_date_input)

# Calculate portfolio value at the start date
portfolio_value_start = portfolio['cash']  # Start with cash value

# Include stock values at the start date
for stock, shares in total_shares_bought.items():
    if start_date in dict_of_stocks[stock].index:
        stock_price = dict_of_stocks[stock].loc[start_date, "Open"]
        portfolio_value_start += stock_price * shares  # Add stock value to portfolio

# Calculate portfolio value at the end date (based on stock prices at the end)
portfolio_value_end = portfolio['cash']

# Include stock values at the end date
for stock, shares in portfolio.items():
    if stock != 'cash':
        if end_date in dict_of_stocks[stock].index:
            stock_price = dict_of_stocks[stock].loc[end_date, "Close"]
        else:
            stock_price = dict_of_stocks[stock]["Close"].loc[:end_date].iloc[-1]

        portfolio_value_end += stock_price * shares


#%%
# Calculate years elapsed
years_elapsed = (end_date - start_date).days / 365.25

# Annualized return (CAGR)
if portfolio_value_start != 0:
    if years_elapsed < 0.1:  # Less than ~1 month of data
        annual_return = (portfolio_value_end / portfolio_value_start) - 1  # Simple return due to no monthly compounding
    else:
        annual_return = (portfolio_value_end / portfolio_value_start) ** (1 / years_elapsed) - 1


print(f"Annual Return (CAGR): {annual_return * 100:.2f}%")

#%% 
# Initialize an empty set to store unique dates
all_dates = set()

# Iterate through the stock names listed in investment_data["Stock"]
for stock in investment_data["Stock"]:
    if stock in dict_of_stocks:
        # Extract the index (Date) of the DataFrame and add to the set
        all_dates.update(dict_of_stocks[stock].index)

# Convert the set to a sorted list
all_dates = sorted(all_dates)

#%%
# Create a dictionary to hold the portfolio value on each day
portfolio_values = {}

# Loop through each day and calculate portfolio value
for date in all_dates:
    portfolio_value = portfolio['cash']
    valid_data = True  # Flag to check if all stocks have data for the date
    
    # Loop through each stock in the portfolio (excluding cash) and calculate its value
    for stock, shares in portfolio.items():
        if stock != 'cash':  # Skip cash
            # Ensure stock data exists for the date
            if date in dict_of_stocks[stock].index:
                stock_price = dict_of_stocks[stock].loc[date, "Close"]
                portfolio_value += shares * stock_price
            else:
                valid_data = False  # Mark as invalid and break out of loop
                break
    
    # Only store the portfolio value if all stocks have valid data
    if valid_data:
        portfolio_values[date] = portfolio_value
    
    # If portfolio_value is 0 despite having cash or stocks, print to debug
    if portfolio_value == 0:
        print(f"Warning: Portfolio value is 0 for {date}. Stocks: {portfolio}, Cash: {portfolio['cash']}")
    
    # Save the portfolio value for the current day
    portfolio_values[date] = portfolio_value
#%%
portfolio_value_df = pd.DataFrame.from_dict(portfolio_values, orient='index', columns=['Portfolio Value'])
portfolio_value_df.index.name = 'Date'
portfolio_value_df = portfolio_value_df.dropna()
portfolio_value_df['Daily Return'] = portfolio_value_df['Portfolio Value'].pct_change()
#%% Finding the portfolio beta

#Select interval for beta ('1d', '1wk', or '1mo' usually work best)
interval = input("\nWhat interval do you want to use to find the portfolio beta? (1d,1wk, or 1mo)")


# Pulls data and adds a pctchange value to the df

portfoliopctchange = portfolio_value_df['Daily Return'].dropna()

#downloads SPDR
spydata = yf.download('SPY', auto_adjust=True, interval = '1d')

spypctchange = spydata['Close'].pct_change()

#Rename column
spypctchange.columns = ['SPY']



#%%
#Cleans up temporary variables
del amount, action, all_dates, stock_price, ticker, total_invested, total_shares_bought, valid_data, valid_dates, shares, portfolio_value, annual_return
#%%
# Input the start date for the 1-year period
date_start1 = today - timedelta(days=365)


# Calculate other start dates and end dates
date_end = end_date
date_start3 = date_start1.replace(year=date_start1.year - 2)
date_start5 = date_start1.replace(year=date_start1.year - 4)


# Convert Series to DataFrames if necessary         
portfoliopctchangedf = portfoliopctchange.to_frame()
portfoliopctchangedf = portfoliopctchangedf.rename(columns={'Daily Return':'Portfolio Return'})

#sets range of dates
spypctchange1 = spypctchange.loc[date_start1:date_end - timedelta(days = 1)]


portfoliopctchange1 = portfoliopctchangedf.loc[date_start1:date_end]


portfoliopctchange3 = portfoliopctchangedf.loc[date_start3:date_end]
spypctchange3 = spypctchange.loc[date_start3:date_end - timedelta(days = 1)]

portfoliopctchange5 = portfoliopctchangedf.loc[date_start5:date_end]
spypctchange5 = spypctchange.loc[date_start5:date_end - timedelta(days = 1)]


#%% Changes interval of returns

if interval == "1d":
    pass

elif interval == "1wk":
    portfoliopctchange1 = portfoliopctchange1.resample('W').apply(lambda x: np.prod(1 + x) - 1)
    portfoliopctchange3 = portfoliopctchange3.resample('W').apply(lambda x: np.prod(1 + x) - 1)
    portfoliopctchange5 = portfoliopctchange5.resample('W').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange1 = spypctchange1.resample('W').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange3 = spypctchange3.resample('W').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange5 = spypctchange5.resample('W').apply(lambda x: np.prod(1 + x) - 1)
elif interval == "1mo":
    portfoliopctchange1 = portfoliopctchange1.resample('M').apply(lambda x: np.prod(1 + x) - 1)
    portfoliopctchange3 = portfoliopctchange3.resample('M').apply(lambda x: np.prod(1 + x) - 1)
    portfoliopctchange5 = portfoliopctchange5.resample('M').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange1 = spypctchange1.resample('M').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange3 = spypctchange3.resample('M').apply(lambda x: np.prod(1 + x) - 1)
    spypctchange5 = spypctchange5.resample('M').apply(lambda x: np.prod(1 + x) - 1)
else:
    raise ValueError("Invalid interval. Use '1d', '1wk', or '1mo'.")

#%%


#makes full df
data1 = spypctchange1.join(portfoliopctchange1)

data3 = spypctchange3.join(portfoliopctchange3)

data5 = spypctchange5.join(portfoliopctchange5)


#%%
# Calculate beta, intercept, and R² for 1-year data
x1 = data1['SPY']
y1 = data1['Portfolio Return']
beta1, intercept1 = np.polyfit(x1, y1, 1)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x1, y1)
r2_1yr = r_value1**2

# Calculate beta, intercept, and R² for 3-year data
x3 = data3['SPY']
y3 = data3['Portfolio Return']
beta3, intercept3 = np.polyfit(x3, y3, 1)
slope3, intercept3, r_value3, p_value3, std_err3 = linregress(x3, y3)
r2_3yr = r_value3**2

# Calculate beta, intercept, and R² for 5-year data
x5 = data5['SPY']
y5 = data5['Portfolio Return']
beta5, intercept5 = np.polyfit(x5, y5, 1)
slope5, intercept5, r_value5, p_value5, std_err5 = linregress(x5, y5)
r2_5yr = r_value5**2

# Print beta and R² values
print(f"1-Year Beta: {beta1:.2f}, R²: {r2_1yr:.2f}")
print(f"3-Year Beta: {beta3:.2f}, R²: {r2_3yr:.2f}")
print(f"5-Year Beta: {beta5:.2f}, R²: {r2_5yr:.2f}")


# Define the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

# Plot 1-Year data
axs[0].scatter(x1, y1, color='blue', alpha=0.6, edgecolor='black', label='Data Points')
axs[0].plot(x1, beta1 * x1 + intercept1, color='red', linewidth=2, label=f'Beta = {beta1:.2f}, R² = {r2_1yr:.2f}')
axs[0].set_xlabel('SPY Returns', fontsize=12)
axs[0].set_ylabel("Portfolio Returns", fontsize=12)
axs[0].set_title('1-Year Regression', fontsize=14, fontweight='bold')
axs[0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].legend(fontsize=12)

# Plot 3-Year data
axs[1].scatter(x3, y3, color='green', alpha=0.6, edgecolor='black', label='Data Points')
axs[1].plot(x3, beta3 * x3 + intercept3, color='red', linewidth=2, label=f'Beta = {beta3:.2f}, R² = {r2_3yr:.2f}')
axs[1].set_xlabel('SPY Returns', fontsize=12)
axs[1].set_ylabel("Portfolio Returns", fontsize=12)
axs[1].set_title('3-Year Regression', fontsize=14, fontweight='bold')
axs[1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].legend(fontsize=12)

# Plot 5-Year data
axs[2].scatter(x5, y5, color='orange', alpha=0.6, edgecolor='black', label='Data Points')
axs[2].plot(x5, beta5 * x5 + intercept5, color='red', linewidth=2, label=f'Beta = {beta5:.2f}, R² = {r2_5yr:.2f}')
axs[2].set_xlabel('SPY Returns', fontsize=12)
axs[2].set_ylabel("Portfolio Returns", fontsize=12)
axs[2].set_title('5-Year Regression', fontsize=14, fontweight='bold')
axs[2].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
axs[2].legend(fontsize=12)

# Add a main title for the figure 
fig.suptitle("Portfolio Beta over 1,3 and 5 years", fontsize=16, fontweight='bold')

# Display the plot
plt.show()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
#%% Cleans up used variables
del data1, data3, data5, beta1, beta3, beta5, x1, x3, x5, y1, y3, y5, std_err1, std_err3, std_err5, p_value1, p_value3, p_value5, years_elapsed, stocks_to_pull
del today, stock, slope1, slope3, slope5, r2_1yr, r2_3yr, r2_5yr, r_value1, r_value3, r_value5, interval, end_date_input, end_date, intercept1,intercept3,intercept5
del date, date_start1, date_start3,date_start5, i, portfolio_value_start, row, stock_data, index, transaction_date, filtered_data
#%%

# Calculate the active return (difference between portfolio and benchmark)
active_return = portfoliopctchange - spypctchange

# Compute tracking error (standard deviation of active return)
tracking_error = np.std(active_return, ddof=1)  # Using ddof=1 for sample standard deviation

print(f"Tracking Error: {tracking_error:.6f}")
#%%
# Load risk-free rate data
risk_free_df = pd.read_csv(r"C:\Users\micha\Desktop\Data (DO NOT MOVE OR CHANGE VARIABLE NAMES)\Risk-Free Rate (Market Yield at 10-year constant maturity.csv", parse_dates=["date"])
risk_free_df.set_index("date", inplace=True)  # Set date as index


# Fill missing risk-free rate data by forward filling
risk_free_df['risk_free_rate'] = risk_free_df['risk_free_rate'].fillna(method='ffill')

# Assuming portfoliopctchange and spypctchange are already Pandas Series with a date index
df = pd.DataFrame({
    "Rp": portfoliopctchange,
    "Rb": spypctchange
})

# Merge with risk-free rate data based on the date index
df = df.join(risk_free_df, how="inner")

# Convert risk-free rate to daily (if it's annualized)
df["Rf"] /= 252  # Adjust if necessary based on your data

# Compute excess returns
df["Excess_Rp"] = df["Rp"] - df["Rf"]
df["Excess_Rb"] = df["Rb"] - df["Rf"]

# Run OLS regression to calculate alpha and beta
X = sm.add_constant(df["Excess_Rb"])  # Independent variable (benchmark excess return)
y = df["Excess_Rp"]  # Dependent variable (portfolio excess return)

model = sm.OLS(y, X).fit()

# Extract alpha and beta
alpha = model.params[0]
beta = model.params[1]

print(f"Alpha: {alpha:.6f}")
print(f"Beta: {beta:.6f}")

#%%

# Calculate mean and standard deviation
mean_return = np.mean(portfoliopctchange)
std_dev = np.std(portfoliopctchange)

# Monte Carlo simulation parameters
num_simulations = 10000
num_days = 252

# Simulate returns
simulated_returns = np.random.normal(loc=mean_return, scale=std_dev, size=(num_days, num_simulations))

simulated_price_paths = np.cumprod(1 + simulated_returns, axis=0)


simulated_price_paths *= portfolio_value_end

# Compute ending values for each simulation
ending_values = simulated_price_paths[-1]

average_path = np.mean(simulated_price_paths, axis=1)

# Calculate percentiles
percentile_10 = np.percentile(ending_values, 10)
percentile_90 = np.percentile(ending_values, 90)
percentile_50 = np.percentile(ending_values, 50)

print(f"10% of the simulations ended below ${percentile_10:,.2f}")
print(f"10% of the simulations ended above ${percentile_90:,.2f}")
print(f"The average return of the simulation was ${percentile_50:,.2f}")
# Plotting
plt.figure(figsize=(14, 7))

# Plot all paths faintly
plt.plot(simulated_price_paths, linewidth=0.5, alpha=0.2)

# Plot average
plt.plot(average_path, color='blue', linewidth=2, label='Average')

# Chart details
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Fama French 5 factor model


