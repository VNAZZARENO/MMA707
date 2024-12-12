# MMA707: Delta Hedging Strategy Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Theory of Delta Hedging](#theory-of-delta-hedging)
    - [Black-Scholes Model](#black-scholes-model)
    - [Delta and Its Role in Hedging](#delta-and-its-role-in-hedging)
    - [Gamma, Theta, and Other Greeks](#gamma-theta-and-other-greeks)
    - [Transaction Costs and Practical Considerations](#transaction-costs-and-practical-considerations)
3. [Implementation](#implementation)
    - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
    - [Volatility Estimation](#volatility-estimation)
    - [Delta Hedging Strategy](#delta-hedging-strategy)
    - [Parameter Sweep and Analysis](#parameter-sweep-and-analysis)
4. [Usage](#usage)
    - [Prerequisites](#prerequisites)
    - [Running the Simulation](#running-the-simulation)
    - [Interpreting the Results](#interpreting-the-results)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Introduction

Delta hedging is a key strategy in options trading to reduce the risk from price changes in the underlying asset. This repository, **MMA707**, implements a delta-hedging approach using the Black-Scholes model, enhanced with volatility estimates from GARCH models and VIX proxies. The goal is to simulate how the hedging strategy performs with different rebalancing and rolling frequencies, while considering transaction costs and other real-world factors.

---

## Theory of Delta Hedging

### Black-Scholes Model

The Black-Scholes model is a widely used method for pricing European-style options. It assumes that the underlying asset price follows a geometric Brownian motion with constant volatility and interest rates. The Black-Scholes formula for a European call option is:

$$
C(S, t) = S \Phi(d_1) - K e^{-r(T-t)} \Phi(d_2)
$$

where:
- $S$ is the current price of the underlying asset.
- $K$ is the strike price of the option.
- $r$ is the risk-free interest rate.
- $T$ is the time to maturity.
- $\Phi(\cdot)$ is the cumulative distribution function of the standard normal distribution.
  
$$
d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T}
$$

### Delta and Its Role in Hedging

Delta $\Delta$ measures how the option price changes with small movements in the underlying asset's price. It’s the first derivative of the option price with respect to the asset price:

$$
\Delta = \frac{\partial C}{\partial S} = \Phi(d_1)
$$

In delta hedging, we aim to create a delta-neutral portfolio, meaning its value doesn't change with small price movements of the underlying asset. For a long call option, delta is positive, so we hedge by shorting $\Delta$ units of the underlying asset.

### Gamma, Theta, and Other Greeks

Other Greeks are also important in options trading and hedging:

- **Gamma $\Gamma$**: Shows how delta changes with the underlying asset's price. A long option position has positive gamma, meaning delta increases as the asset price rises. High gamma requires more frequent rebalancing to stay delta-neutral.
  
- **Theta $\Theta$**: Represents time decay, or how the option's price decreases as time passes. For long options, theta is negative, indicating the option loses value over time.

- **Vega $\nu$**: Measures sensitivity to volatility changes.

Understanding these Greeks helps manage the risks and effectiveness of hedging strategies.

### Transaction Costs and Practical Considerations

In theory, continuous delta hedging can perfectly replicate option payoffs. However, in reality, several issues arise:

- **Transaction Costs**: Rebalancing the portfolio incurs costs like spreads, commissions, and slippage. Frequent rebalancing, especially with high gamma, can add up these costs quickly.
  
- **Discrete Rebalancing**: Continuous rebalancing isn't feasible. Rebalancing at set intervals (e.g., daily or weekly) can cause the portfolio's delta to drift between adjustments.
  
- **Volatility Estimation**: Accurately estimating volatility is tough. Errors in estimation lead to incorrect delta calculations, affecting the hedge.
  
- **Model Assumptions**: The Black-Scholes model assumes constant volatility and interest rates, which isn't always true in real markets.

These factors often result in a delta-hedged long option position losing value over time, as shown in our simulations.

---

## Implementation

### Data Collection and Preprocessing

The simulation uses historical data for the underlying asset, risk-free rates, and volatility proxies like the VIX. Data is gathered using the yfinance library and processed as follows:

- **Stock Data**: Historical closing prices are used to calculate log returns.
  
- **Risk-Free Rate**: Sourced from a relevant financial instrument (e.g., 5-Year Treasury Rate) and annualized.
  
- **Volatility Proxy**: VIX or other implied volatility measures serve as proxies for the asset's volatility.

### Volatility Estimation

We estimate volatility using two methods:

1. **GARCH Model**: The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model captures changing volatility by modeling the conditional variance of returns. This provides a smooth estimate of annualized volatility for the hedging strategy.
  
2. **VIX Proxy**: Implied volatility from market indices like the VIX offers an alternative volatility estimate.

### Delta Hedging Strategy

Key parts of the delta hedging simulation include:

- **Initial Positioning**: Start by buying a call option and hedging by shorting the underlying asset based on the option's delta.
  
- **Simulation of Trades**: Rebalance the portfolio at set intervals (day_rebalancing) and roll the option position (day_rolling) to maintain the hedge. This process accounts for transaction costs and time decay.
  
- **Parameter Sweep**: Test different combinations of rebalancing (day_stock) and rolling (day_option) frequencies to see how they affect the portfolio's final value.

### Parameter Sweep and Analysis

To check the strategy's robustness, the simulation runs through various values of day_stock (rebalancing frequency) and day_option (rolling frequency). For each pair, the final portfolio value is recorded and visualized with heatmaps to find the best rebalancing and rolling schedules.

---

## Usage

### Prerequisites

Make sure you have these Python libraries installed:

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- yfinance
- arch
- tqdm

Install them with pip:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance arch tqdm
```

### Running the Simulation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/VNAZZARENO/MMA707.git
    cd MMA707
    ```

2. **Configure Parameters:**

   Adjust settings in the main script, such as the underlying symbol, risk-free rate proxy, volatility proxy, and the historical data period.

3. **Open the Notebook and Modify as Needed:**

   Tinker with the notebook to customize the simulation.

4. **View Results:**

   The simulation will create CSV files with portfolio data and generate heatmaps showing final portfolio values for different parameter combinations.

### Interpreting the Results

- **Final Portfolio Value:** Shows how profitable the delta hedging strategy is after considering transaction costs and other factors.
  
- **Heatmaps:** Visualize how different rebalancing and rolling frequencies impact the strategy's performance. Look for parameters that maximize the final portfolio value while keeping costs low.
  
- **Cumulative Fees and Delta Drift:** Offer insights into the costs incurred and how well the portfolio maintains delta neutrality over time.

---

## Conclusion

The delta hedging strategy in this repository highlights the real-world challenges of replicating option payoffs. While the Black-Scholes model suggests that continuous hedging can perfectly match option payoffs, practical issues like transaction costs, discrete rebalancing, and volatility estimation errors lead to a gradual loss in portfolio value. Our parameter sweep analysis shows the balance between hedging frequency and transaction costs, emphasizing the need to optimize rebalancing schedules to reduce losses.

---

## References

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
- Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics and Management Science, 4(1), 141-183.
- Leland, H. E. (1985). *Option Pricing and Replication with Transactions Costs*. Journal of Finance, 40(3), 667-685.
- Boyle, P., & Vorst, T. (1992). *Option Replication in Discrete Time with Transaction Costs*. Journal of Finance, 47(5), 1731-1751.
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [ARCH Package Documentation](https://arch.readthedocs.io/en/latest/)
