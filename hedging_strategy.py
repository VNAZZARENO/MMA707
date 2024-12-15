import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from tick.hawkes import SimuHawkes, HawkesKernelExp

st.set_page_config(layout="wide")

###################################
# Functions for Synthetic Data
###################################
def generate_gbm_prices(start_price, mu, sigma, days):
    dt = 1/252  
    random_returns = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt,
                                      scale=sigma * np.sqrt(dt),
                                      size=days)
    price_series = start_price * np.exp(np.cumsum(random_returns))
    return pd.Series(price_series)

def generate_risk_free_rates(rate, days):
    daily_rate = rate / 252
    return pd.Series([daily_rate] * days)

def generate_hawkes_volatility(days, dt=1/252, mu=0.8, alpha=0.6, beta=0.7, jump_size=0.04, seed=123):
    kernel = HawkesKernelExp(decay=beta, intensity=alpha)
    kernels = np.array([[kernel]])
    baseline = [mu]

    hawkes = SimuHawkes(kernels=kernels, baseline=baseline, end_time=days, seed=seed, verbose=False)
    hawkes.simulate()
    event_times = hawkes.timestamps[0]

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(days * 1.5))
    dates = pd.bdate_range(start=start_date, periods=days, freq='C')
    volatility = np.zeros(len(dates))
    decay_factor = np.exp(-beta * dt)

    event_days = np.floor(event_times).astype(int)
    for day in event_days:
        if day < len(volatility):
            decay = decay_factor ** np.arange(len(volatility) - day)
            volatility[day:] += jump_size * decay[:len(volatility) - day]

    volatility += 0.2  
    volatility_series = pd.Series(volatility, index=dates)
    return volatility_series

def get_synthetic_data_hawkes(time_period=1, 
                              start_price=100,
                              mu=0.08,
                              sigma=0.25,
                              rf_rate=0.02,
                              hawkes_mu=0.8,
                              hawkes_alpha=0.6,
                              hawkes_beta=0.7,
                              hawkes_jump=0.04,
                              seed=42):

    total_days = int(time_period * 252)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(total_days * 1.5))
    dates = pd.bdate_range(start=start_date, periods=total_days, freq='C')

    stock_prices = generate_gbm_prices(start_price, mu, sigma, total_days)
    stock_prices.index = dates
    log_returns = np.log(stock_prices / stock_prices.shift(1))
    stock_data = pd.DataFrame({
        'Close': stock_prices,
        'log_returns': log_returns
    })
    stock_data.dropna(subset=['log_returns'], inplace=True)

    risk_free_data = generate_risk_free_rates(rf_rate, len(stock_data)) * 100
    risk_free_df = pd.DataFrame({'Rate': risk_free_data.values}, index=stock_data.index)

    volatility_proxy_data = generate_hawkes_volatility(
        days=len(stock_data),
        mu=hawkes_mu,
        alpha=hawkes_alpha,
        beta=hawkes_beta,
        jump_size=hawkes_jump,
        seed=seed
    )
    volatility_proxy_data.name = 'Vol_Proxy'
    stock_data['implied_volatility'] = volatility_proxy_data.values
    data = stock_data.copy()
    data = pd.merge(data, risk_free_df, how='left', left_index=True, right_index=True)
    data.ffill(inplace=True)

    garch_model = arch_model(data['log_returns'] * 100, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
    garch_fit = garch_model.fit(disp="off")
    data['volatility'] = garch_fit.conditional_volatility
    data['annualized_volatility'] = data['volatility'] * np.sqrt(252) / 100
    data['smoothed_annualized_volatility'] = data['annualized_volatility'].rolling(window=3, min_periods=1).mean()
    data.dropna(subset=['smoothed_annualized_volatility'], inplace=True)
    return data

###################################
# Black-Scholes and Hedging Functions
###################################
def compute_d1_d2(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes(S, K, T, r=0.02, sigma=0.2, option_type='call'):
    d1, d2 = compute_d1_d2(S, K, r=r, sigma=sigma, T=T)
    if option_type == 'call':
        c = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put': 
        c = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Error, option must be 'call' or 'put'")
    return c, norm.cdf(d1)

###################################
# Portfolio Initialization
###################################
def build_initial_positions(data, fees=1/100, K_multiplier=1.1, use_vix_proxy=False, use_constant_rate=False, risk_free=0.02, option_maturity=6/12):
    N_call = 1
    S0 = data['Close'].iloc[0]
    K0 = int(round(K_multiplier * S0, -1))
    if use_vix_proxy:
        g0 = data['implied_volatility'].iloc[0]
    else:
        g0 = data['annualized_volatility'].iloc[0]

    if use_constant_rate:
        r0 = risk_free
    else:
        r0 = data['Rate'].iloc[0]

    T0 = option_maturity
    call_price, call_delta = black_scholes(S0, K0, T0, r0, g0, 'call')
    shares = N_call * call_delta
    bank = (-N_call * call_price) + ((shares) * S0)
    initial_fees = (N_call * call_price + abs(shares * S0)) * fees
    bank -= initial_fees
    value_portfolio = bank + (- shares * S0) + (N_call * call_price)

    return {
        "S": S0,
        "K": K0,
        "T": T0,
        "Volatility": g0,
        "Risk Free": r0, 
        "Shares": -shares,
        "Share_Price": S0 * shares,
        "Option_Type": 'call',
        "Option_Price": call_price,
        "Option_Delta": call_delta,
        "Bank": bank,
        "Value_portfolio": value_portfolio,
        "fees_cumsum": initial_fees,
        "fees_transaction": initial_fees
    }

###################################
# Simulation of Trades
###################################
def simulate_trades(data, initial_positions, fees=1/100, day_rebalancing=1, day_rolling=21, K_multiplier=1.1, use_vix_proxy=False, use_constant_rate=False, risk_free=0.02, option_maturity=6/12):
    portfolio = {}
    position = initial_positions.copy()
    cumulative_drift = 0  
    portfolio[data.index[0]] = position
    N_call = 1 

    for i in range(day_rebalancing, len(data), day_rebalancing):
        prev_position = portfolio[data.index[i - day_rebalancing]].copy()
        S = data['Close'].iloc[i]
        K = prev_position['K']
        if use_vix_proxy:
            g = data['implied_volatility'].iloc[i]
        else:
            g = data['smoothed_annualized_volatility'].iloc[i]

        if use_constant_rate:
            r = risk_free
        else:
            r = data['Rate'].iloc[i]

        T = prev_position['T'] - (day_rebalancing / 252)
        if T <= 0:
            # Option expired
            payoff = np.maximum(S-K,0)
            K = int(round(K_multiplier * S, -1))
            call_price, call_delta = black_scholes(S, K, option_maturity, r, g, 'call')

            shares_old = prev_position['Shares']
            shares_new = N_call * call_delta
            shares_change = - shares_new - abs(shares_old)

            transaction_fee = call_price * fees
            bank = prev_position['Bank'] * np.exp(r * (day_rebalancing / 252))
            bank -= transaction_fee
            bank += payoff

            transaction_fee = abs(shares_change * S) * fees
            bank -= transaction_fee

            fees_cumsum = prev_position['fees_cumsum'] + transaction_fee
            value_portfolio = bank + (-shares_new * S) + (N_call * call_price)

            total_delta = call_delta - shares_new
            cumulative_drift += abs(total_delta)

            new_position = {
                "S": S,
                "K": K,
                "T": T,
                "Volatility": g,
                "Risk Free": r,
                "Shares": -shares_new,
                "Share_Price": S * shares_new,
                "Option_Type": 'call',
                "Option_Price": call_price,
                "Option_Delta": call_delta,
                "Bank": bank,
                "Value_portfolio": value_portfolio,
                "fees_transaction": transaction_fee,
                "fees_cumsum": fees_cumsum,
                "Cumulative_Delta_Drift": cumulative_drift,
                "day_rolling": day_rolling
            }

            portfolio[data.index[i]] = new_position

        else:
            if (i % day_rolling == 0):
                # Roll option
                call_price, call_delta = black_scholes(S, K, T, r, g, 'call')
                shares_old = prev_position['Shares']
                shares_new = N_call * call_delta
                shares_change = - shares_new - shares_old

                bank = prev_position['Bank'] * np.exp(r * (day_rebalancing / 252))
                bank += - call_price * fees
                bank += call_price

                T = option_maturity
                S = data['Close'].iloc[i]
                K = int(round(K_multiplier * S, -1))

                transaction_fee = call_price * fees
                bank -= transaction_fee
                bank -= call_price

                fees_cumsum = prev_position['fees_cumsum'] + transaction_fee
                value_portfolio = bank + (-shares_new * S) + (N_call * call_price)

                total_delta = call_delta - shares_new
                cumulative_drift += abs(total_delta)

                new_position = {
                    "S": S,
                    "K": K,
                    "T": T,
                    "Volatility": g,
                    "Risk Free": r,
                    "Shares": -shares_new,
                    "Share_Price": S * shares_new,
                    "Option_Type": 'call',
                    "Option_Price": call_price,
                    "Option_Delta": call_delta,
                    "Bank": bank,
                    "Value_portfolio": value_portfolio,
                    "fees_transaction": transaction_fee,
                    "fees_cumsum": fees_cumsum,
                    "Cumulative_Delta_Drift": cumulative_drift,
                    "day_rolling": day_rolling
                }
                portfolio[data.index[i]] = new_position

            else:
                # Dynamic adjustment of rolling frequency based on vol regime
                if day_rolling > 0:
                    window_length = 21
                    if i >= window_length:
                        if use_vix_proxy:
                            rolling_g = data['implied_volatility'].iloc[i - window_length:i].mean()
                        else:
                            rolling_g = data['smoothed_annualized_volatility'].iloc[i - window_length:i].mean()

                        if g > 2 * rolling_g:
                            day_rolling = max(1, int(day_rolling // 1.1))
                        else:
                            day_rolling = int(min(252, day_rolling * 1.04))

                bank = prev_position['Bank'] * np.exp(r * (day_rebalancing / 252))
                call_price, call_delta = black_scholes(S, K, T, r, g, 'call')

                shares_old = prev_position['Shares']
                shares_new = N_call * call_delta
                shares_change = - shares_new - shares_old

                transaction_fee = abs(shares_change * S) * fees
                bank -= transaction_fee
                fees_cumsum = prev_position['fees_cumsum'] + transaction_fee
                value_portfolio = bank + (-shares_new * S) + (N_call * call_price)

                total_delta = call_delta - shares_new
                cumulative_drift += abs(total_delta)

                new_position = {
                    "S": S,
                    "K": K,
                    "T": T,
                    "Volatility": g,
                    "Risk Free": r,
                    "Shares": -shares_new,
                    "Share_Price": S * shares_new,
                    "Option_Type": 'call',
                    "Option_Price": call_price,
                    "Option_Delta": call_delta,
                    "Bank": bank,
                    "Value_portfolio": value_portfolio,
                    "fees_transaction": transaction_fee,
                    "fees_cumsum": fees_cumsum,
                    "Cumulative_Delta_Drift": cumulative_drift,
                    "day_rolling": day_rolling
                }

                portfolio[data.index[i]] = new_position

    results = pd.DataFrame(portfolio).T
    results['Date'] = pd.to_datetime(results.index)
    return results

###################################
# Streamlit UI
###################################

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", ["Simulation", "Volatility Visualization", "Hedging Computations", "Positions Overview"])

st.sidebar.header("Parameters")

data_source = st.sidebar.selectbox("Data Source", ["Real Market Data", "Synthetic (Hawkes)"])
stock_ticker = st.sidebar.text_input("Stock Ticker (For Real Data)", value="SPY")
year_history = st.sidebar.slider("History (years)", 0.5, 20.0, 1.0, 0.5)

start_date = datetime.date.today() - datetime.timedelta(days=int(year_history*365)) 
end_date = datetime.date.today()

day_rebalancing = st.sidebar.number_input("Rebalancing Frequency (days)",1,252,1)
day_rolling = st.sidebar.number_input("Rolling Frequency (days)",-1,1000,21)
fees = st.sidebar.slider("Transaction Fees (in %)", 0.0, 5.0, 1.0)/100.0
option_maturity = st.sidebar.slider("Option Maturity (years)", 0.1, 2.0, 0.5, 0.1)
use_vix_proxy = st.sidebar.selectbox("Volatility Source", ["GARCH","VIX"]) == "VIX"
K_multiplier = st.sidebar.slider("Strike Multiplier", 0.8, 1.2, 1.1, 0.05)
run_button = st.sidebar.button("Run Simulation")

if run_button:
    if data_source == "Real Market Data":
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
        if len(stock_data) == 0:
            st.error("No data found for the given ticker and date range.")
        else:
            stock_data['log_returns'] = np.log(stock_data['Close']/stock_data['Close'].shift(1))
            stock_data.dropna(subset=['log_returns'], inplace=True)
            risk_free_data = yf.download("^FVX", start=start_date, end=end_date)[['Close']]/100
            risk_free_data.columns=["Rate"]
            vix_data=yf.download("^VIX", start=start_date, end=end_date)[['Close']]
            vix_data.columns=['VIX']
            vix_data['implied_volatility']=vix_data['VIX']/100.0
            data=pd.merge(stock_data,risk_free_data,how='left',on='Date')
            data=pd.merge(data,vix_data[['implied_volatility']],how='left',on='Date')
            data.ffill(inplace=True)
            garch_model=arch_model(data['log_returns']*100, vol='Garch', p=1,q=1,mean='Zero', dist='normal')
            garch_fit=garch_model.fit(disp="off")
            data['volatility']=garch_fit.conditional_volatility
            data['annualized_volatility']=data['volatility']*np.sqrt(252)/100
            data['smoothed_annualized_volatility']=data['annualized_volatility'].rolling(window=3,min_periods=1).mean()
            data.dropna(subset=['smoothed_annualized_volatility'], inplace=True)
    else:
        # Synthetic Data
        # start_price from the first close of real data if available or 100 as default
        start_price = 100
        data = get_synthetic_data_hawkes(
            time_period=year_history,
            start_price=start_price,
            mu=0.08,
            sigma=0.25,
            rf_rate=0.02,
            hawkes_mu=0.8,
            hawkes_alpha=0.6,
            hawkes_beta=0.7,
            hawkes_jump=0.04,
            seed=42
        )

    if len(data) > 0:
        initial_positions=build_initial_positions(data=data,fees=fees,use_vix_proxy=use_vix_proxy,option_maturity=option_maturity,K_multiplier=K_multiplier)
        portfolio_data=simulate_trades(data, initial_positions, fees=fees, day_rebalancing=day_rebalancing, day_rolling=day_rolling, use_vix_proxy=use_vix_proxy, option_maturity=option_maturity, K_multiplier=K_multiplier)
        st.session_state['data']=data
        st.session_state['portfolio_data']=portfolio_data
    else:
        st.write("Please select valid parameters and run again.")

if page == "Simulation":
    if 'portfolio_data' in st.session_state:
        st.subheader("Simulation Results")
        portfolio_data = st.session_state['portfolio_data']
        
        # Show dataframe
        numeric_cols = portfolio_data.select_dtypes(include=[float, int]).columns
        st.dataframe(portfolio_data.style.format({col: "{:.4f}" for col in numeric_cols}))

        # First Figure: Portfolio Value and Volatility
        # We'll use Plotly and add a secondary y-axis for volatility
        fig_val = go.Figure()
        fig_val.add_trace(
            go.Scatter(
                x=portfolio_data['Date'],
                y=portfolio_data['Value_portfolio'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green')
            )
        )

        # Add volatility on a secondary axis, dashed line, alpha=0.3 equivalent is opacity=0.3
        # Make sure 'Volatility' is in portfolio_data. If not, use annualized vol or another vol measure.
        if 'Volatility' in portfolio_data.columns:
            fig_val.add_trace(
                go.Scatter(
                    x=portfolio_data['Date'],
                    y=portfolio_data['Volatility'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='blue', dash='dash'), 
                    opacity=0.3,
                    yaxis="y2"
                )
            )

        fig_val.update_layout(
            title="Portfolio Value and Volatility Over Time",
            xaxis_title='Date',
            yaxis=dict(title='Portfolio Value', side='left'),
            yaxis2=dict(
                title='Volatility',
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0, y=1.1, orientation='h')
        )

        st.plotly_chart(fig_val, use_container_width=True)

        # Second Figure: Option Price and Delta
        # We'll plot Option_Price on the left axis and Option_Delta on the right axis
        if 'Option_Price' in portfolio_data.columns and 'Option_Delta' in portfolio_data.columns:
            fig_opt = go.Figure()
            fig_opt.add_trace(
                go.Scatter(
                    x=portfolio_data['Date'],
                    y=portfolio_data['Option_Price'],
                    mode='lines',
                    name='Option Price',
                    line=dict(color='purple')
                )
            )

            # Add Delta on secondary axis
            fig_opt.add_trace(
                go.Scatter(
                    x=portfolio_data['Date'],
                    y=portfolio_data['Option_Delta'],
                    mode='lines',
                    name='Option Delta',
                    line=dict(color='blue'),
                    yaxis='y2'
                )
            )

            fig_opt.update_layout(
                title="Option Price and Delta (Delta Hedge)",
                xaxis_title='Date',
                yaxis=dict(title='Option Price', color='purple'),
                yaxis2=dict(
                    title='Delta',
                    overlaying='y',
                    side='right',
                    color='blue'
                ),
                legend=dict(x=0, y=1.1, orientation='h')
            )

            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.write("Option price or delta data not found in the portfolio_data. Please ensure these columns exist.")

    else:
        st.write("Please run the simulation first.")


elif page=="Volatility Visualization":
    if 'data' in st.session_state:
        data=st.session_state['data']
        st.subheader("Volatility Visualization")
        if 'implied_volatility' not in data.columns:
            st.write("No implied_volatility in dataset.")
        else:
            st.write("Below we compare GARCH-based annualized volatility estimates to the implied volatility (or synthetic VIX).")
            fig_vol=go.Figure()
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['smoothed_annualized_volatility'], mode='lines', name='GARCH Smoothed Ann. Vol', line=dict(color='red')))
            if 'implied_volatility' in data.columns:
                fig_vol.add_trace(go.Scatter(x=data.index, y=data['implied_volatility'], mode='lines', name='Implied Vol', line=dict(color='blue')))
            fig_vol.update_layout(title="Volatility Comparison", xaxis_title='Date', yaxis_title='Volatility')
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.write("Please run the simulation first.")

elif page=="Hedging Computations":
    st.subheader("Hedging Computations and Formulas")
    st.markdown("### Black–Scholes Pricing Formula")
    st.latex(r"""
    C(S,t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
    """)
    st.markdown("where")
    st.latex(r"""
    d_1 = \frac{\ln(\frac{S_t}{K}) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}}, \quad
    d_2 = d_1 - \sigma \sqrt{T-t}.
    """)
    st.markdown("### Delta and Gamma")
    st.latex(r"\Delta = N(d_1)")
    st.latex(r"\Gamma = \frac{N'(d_1)}{S_t \sigma \sqrt{T-t}}")
    st.markdown("**Delta Hedging:** To delta-hedge a call, you short Δ units of the underlying. This removes first-order price risk.")
    st.markdown("**Gamma Hedging:** By also trading options (puts/calls), you can hedge gamma, removing second-order price risk (locally).")

elif page == "Positions Overview":
    if 'portfolio_data' in st.session_state:
        portfolio_data = st.session_state['portfolio_data']
        st.subheader("Detailed Portfolio Positions Over Time")
        st.markdown("### Positions Table")
        numeric_cols = portfolio_data.select_dtypes(include=[float, int]).columns
        st.dataframe(portfolio_data.style.format({col: "{:.4f}" for col in numeric_cols}))


        fig_positions = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                     subplot_titles=("Shares Held Over Time", "Bank Balance Over Time", "Value Portfolio Over Time"))
        fig_positions.add_trace(go.Scatter(x=portfolio_data['Date'], y=portfolio_data['Shares'],
                                           mode='lines', name='Shares Held', line=dict(color='blue')),
                                 row=1, col=1)
        fig_positions.add_trace(go.Scatter(x=portfolio_data['Date'], y=portfolio_data['Bank'],
                                           mode='lines', name='Bank Balance', line=dict(color='green')),
                                 row=2, col=1)
        fig_positions.add_trace(go.Scatter(x=portfolio_data['Date'], y=portfolio_data['Value_portfolio'],
                                           mode='lines', name='Portfolio Value', line=dict(color='purple')),
                                 row=3, col=1)
        fig_positions.update_layout(height=900, width=1200, title_text="Positions Over Time",
                                    showlegend=False)
        fig_positions.update_xaxes(title_text="Date", row=3, col=1)
        fig_positions.update_yaxes(title_text="Shares", row=1, col=1)
        fig_positions.update_yaxes(title_text="Bank ($)", row=2, col=1)
        fig_positions.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
        st.plotly_chart(fig_positions, use_container_width=True)
    else:
        st.write("Please run the simulation first.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("#### For questions : vincent.nazzareno@gmail.com", unsafe_allow_html=True)
