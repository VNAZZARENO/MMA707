import streamlit as st
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots




st.set_page_config(layout="wide")


####################################
# Fetching Data Function
####################################

@st.cache_data(show_spinner=True, ttl=86400)  # Cache expires after 1 day (86400 seconds)
def get_data(symbol, risk_free, volatility_proxy, time_period):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(time_period * 252))
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Remove the symbol prefix because we only retrieve one stock
    if isinstance(stock_data.columns, pd.MultiIndex):  # Check for MultiIndex
        stock_data.columns = stock_data.columns.droplevel(1)
    if stock_data.empty:
        st.error(f"No data found for symbol: {symbol}")
        return pd.DataFrame()

    stock_data['log_returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    stock_data.dropna(subset=['log_returns'], inplace=True)

    # Fetch risk-free rate data
    risk_free_data = yf.download(risk_free, start=start_date, end=end_date)
    if risk_free_data.empty:
        st.error(f"No data found for risk-free symbol: {risk_free}")
        return pd.DataFrame()
    risk_free_data = risk_free_data[['Close']] / 100
    risk_free_data.columns = ["Rate"]

    # Initialize volatility proxy
    if volatility_proxy == "VIX":
        vol_proxy_symbol = "^VIX"
        volatility_data = yf.download(vol_proxy_symbol, start=start_date, end=end_date)
        if isinstance(volatility_data.columns, pd.MultiIndex):  # Check for MultiIndex
            volatility_data.columns = volatility_data.columns.droplevel(1)
        if volatility_data.empty:
            st.error(f"No data found for volatility proxy symbol: {vol_proxy_symbol}")
            return pd.DataFrame()
        volatility_data = volatility_data[['Close']].rename(columns={'Close': 'implied_volatility'})
        volatility_data['implied_volatility'] = volatility_data['implied_volatility'] / 100.0

    elif volatility_proxy == "VXN":
        vol_proxy_symbol = "^VXN"
        volatility_data = yf.download(vol_proxy_symbol, start=start_date, end=end_date)
        if isinstance(volatility_data.columns, pd.MultiIndex):  # Check for MultiIndex
            volatility_data.columns = volatility_data.columns.droplevel(1)
        if volatility_data.empty:
            st.error(f"No data found for volatility proxy symbol: {vol_proxy_symbol}")
            return pd.DataFrame()
        volatility_data = volatility_data[['Close']].rename(columns={'Close': 'implied_volatility'})
        volatility_data['implied_volatility'] = volatility_data['implied_volatility'] / 100.0

    elif volatility_proxy == "Realized Volatility":
        # Calculate realized volatility
        volatility_data = pd.DataFrame()
        volatility_data['realized_volatility'] = stock_data['log_returns'].rolling(window=21).std() * np.sqrt(252)

    elif volatility_proxy == "EWMA":
        lambda_ = 0.94
        volatility_data = pd.DataFrame()
        volatility_data['ewma_volatility'] = stock_data['log_returns'].ewm(alpha=1 - lambda_).std() * np.sqrt(252)

    elif volatility_proxy == "ATR":
        # Calculate ATR-based volatility
        volatility_data = pd.DataFrame()
        volatility_data['ATR'] = stock_data[['High', 'Low', 'Close']].apply(
            lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])),
            axis=1
        )
        volatility_data['atr_volatility'] = volatility_data['ATR'].rolling(window=14).mean() * np.sqrt(252) / stock_data['Close']

    elif volatility_proxy == "GARCH":
        # Estimate volatility using GARCH
        garch_model = arch_model(stock_data['log_returns'] * 100, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
        garch_fit = garch_model.fit(disp="off")
        stock_data['volatility'] = garch_fit.conditional_volatility
        stock_data['annualized_volatility'] = stock_data['volatility'] * np.sqrt(252) / 100
        stock_data['smoothed_annualized_volatility'] = stock_data['annualized_volatility'].rolling(window=3, min_periods=1).mean()
        volatility_data = stock_data[['smoothed_annualized_volatility']].rename(columns={'smoothed_annualized_volatility': 'implied_volatility'})

    else:
        # No volatility proxy provided
        volatility_data = pd.DataFrame()

    # Merge data
    if not volatility_data.empty:
        data = stock_data.join(risk_free_data, how='left').join(volatility_data, how='left')
    else:
        data = stock_data.join(risk_free_data, how='left')

    data.ffill(inplace=True)

    # If using GARCH separately
    if volatility_proxy != "GARCH":
        # Estimate GARCH volatility regardless of the proxy
        garch_model = arch_model(data['log_returns'] * 100, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
        garch_fit = garch_model.fit(disp="off")
        data['volatility'] = garch_fit.conditional_volatility
        data['annualized_volatility'] = data['volatility'] * np.sqrt(252) / 100
        data['smoothed_annualized_volatility'] = data['annualized_volatility'].rolling(window=3, min_periods=1).mean()

    data.dropna(subset=['log_returns'], inplace=True)

    return data



def calculate_beta(stock_returns, benchmark_returns):
    covariance = np.cov(stock_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance
    return beta




####################################
#Black-Scholes and Hedging Functions
####################################
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

@st.cache_data(show_spinner=True)
def build_initial_positions(data, fees=1/100, K_multiplier=1.1, use_vix_proxy=False, use_constant_rate=False, risk_free=0.02, use_realized_vol=False, use_ewma=False, use_atr=False, option_maturity=6/12):
    N_call = 1
    S0 = data['Close'].iloc[0]
    K0 = int(round(K_multiplier * S0, -1))
    if use_vix_proxy:
        g0 = data['implied_volatility'].iloc[0]
    elif use_realized_vol:
        g0 = data['realized_volatility'].iloc[0]
    elif use_ewma:
        g0 = data['ewma_volatility'].iloc[0]
    elif use_atr:
        g0 = data['atr_volatility'].iloc[0]
    else:
        g0 = data['smoothed_annualized_volatility'].iloc[0]


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


    # Similar adjustments as in build_initial_positions to select the correct volatility measure
    # Example:
    

    # Continue with existing simulation logic...


@st.cache_data(show_spinner=True)
def simulate_trades(data, initial_positions, fees=1/100, day_rebalancing=1, day_rolling=21, 
                   use_vix_proxy=False, use_constant_rate=False, use_realized_vol=False, use_ewma=False, use_atr=False, 
                   risk_free=0.02, option_maturity=6/12, K_multiplier=1.1):
                   
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
        elif use_realized_vol:
            g = data['realized_volatility'].iloc[i]
        elif use_ewma:
            g = data['ewma_volatility'].iloc[i]
        elif use_atr:
            g = data['atr_volatility'].iloc[i]
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

stock_ticker = st.sidebar.text_input("Stock Ticker (For Real Data)", value="SPY")
year_history = st.sidebar.slider("History (years)", 0.5, 30.0, 1.0, 0.5)

# start_date = datetime.date.today() - datetime.timedelta(days=int(year_history*252)) 
# end_date = datetime.date.today()

day_rebalancing = st.sidebar.number_input("Rebalancing Frequency (days)",1,252,1)
day_rolling = st.sidebar.number_input("Rolling Frequency (days)",-1,1000,21)
fees = st.sidebar.slider("Transaction Fees (in %)", 0.0, 5.0, 1.0)/100.0
option_maturity = st.sidebar.slider("Option Maturity (years)", 0.1, 2.0, 0.5, 0.1)
volatility_proxy = st.sidebar.selectbox("Volatility Source", ["GARCH", "VIX", "VXN", "Realized Volatility", "EWMA", "ATR"])
K_multiplier = st.sidebar.slider("Strike Multiplier", 0.5, 1.5, 1.1, 0.05)
run_button = st.sidebar.button("Run Simulation")

if run_button:
    st.write(volatility_proxy)
    data = get_data(stock_ticker, "^FVX", volatility_proxy, year_history)
    if not data.empty:
        # Determine if VIX or other proxies are used
        use_vix_proxy = (volatility_proxy == "VIX") or (volatility_proxy == "VXN")
        use_realized_vol = (volatility_proxy == "Realized Volatility")
        use_ewma = (volatility_proxy == "EWMA")
        use_atr = (volatility_proxy == "ATR")

        initial_positions = build_initial_positions(
            data=data,
            fees=fees,
            use_vix_proxy=use_vix_proxy,
            use_realized_vol=use_realized_vol,
            use_ewma=use_ewma,
            use_atr=use_atr,
            option_maturity=option_maturity,
            K_multiplier=K_multiplier
        )

        portfolio_data = simulate_trades(
            data,
            initial_positions,
            fees=fees,
            day_rebalancing=day_rebalancing,
            day_rolling=day_rolling,
            use_vix_proxy=use_vix_proxy,
            use_realized_vol=use_realized_vol,
            use_ewma=use_ewma,
            use_atr=use_atr,
            option_maturity=option_maturity,
            K_multiplier=K_multiplier
        )

        st.session_state['data'] = data
        st.session_state['portfolio_data'] = portfolio_data
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


if page == "Volatility Visualization":
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.subheader("Volatility Visualization")
        fig_vol = go.Figure()
        
        # Plot the selected volatility proxy
        if volatility_proxy == "GARCH":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['smoothed_annualized_volatility'], 
                                         mode='lines', name='GARCH Smoothed Ann. Vol', line=dict(color='red')))
        elif volatility_proxy == "VIX":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['implied_volatility'], 
                                         mode='lines', name='VIX', line=dict(color='blue')))
        elif volatility_proxy == "VXN":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['implied_volatility'], 
                                         mode='lines', name='VXN', line=dict(color='orange')))
        elif volatility_proxy == "Realized Volatility":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['realized_volatility'], 
                                         mode='lines', name='Realized Volatility', line=dict(color='green')))
        elif volatility_proxy == "EWMA":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['ewma_volatility'], 
                                         mode='lines', name='EWMA Volatility', line=dict(color='purple')))
        elif volatility_proxy == "ATR":
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['atr_volatility'], 
                                         mode='lines', name='ATR Volatility', line=dict(color='brown')))
        
        # Optionally, plot other volatility measures for comparison
        if volatility_proxy != "GARCH" and "smoothed_annualized_volatility" in data.columns:
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['smoothed_annualized_volatility'], 
                                         mode='lines', name='GARCH Smoothed Ann. Vol', 
                                         line=dict(color='red', dash='dash')))
        
        fig_vol.update_layout(title="Volatility Comparison", 
                              xaxis_title='Date', 
                              yaxis_title='Volatility')
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
