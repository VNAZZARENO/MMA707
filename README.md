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

Delta hedging is a fundamental strategy in options trading aimed at minimizing the directional risk associated with price movements of the underlying asset. This repository, **MMA707**, provides an implementation of a delta-hedging scheme using the Black-Scholes model, augmented with volatility estimates from both GARCH models and VIX proxies. The objective is to simulate the performance of the hedging strategy under various rebalancing and rolling frequencies, accounting for transaction costs and other practical considerations.

---

## Theory of Delta Hedging

### Black-Scholes Model

The Black-Scholes model is a mathematical framework for pricing European-style options. It assumes that the price of the underlying asset follows a geometric Brownian motion with constant volatility and interest rates. The Black-Scholes formula for a European call option is given by:

\[
C(S, t) = S \Phi(d_1) - K e^{-r(T-t)} \Phi(d_2)
\]

where:
- \( S \) is the current price of the underlying asset.
- \( K \) is the strike price of the option.
- \( r \) is the risk-free interest rate.
- \( T \) is the time to maturity.
- \( \Phi(\cdot) \) is the cumulative distribution function of the standard normal distribution.
- \( d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}} \)
- \( d_2 = d_1 - \sigma \sqrt{T} \)

### Delta and Its Role in Hedging

Delta (\( \Delta \)) represents the sensitivity of the option's price to small changes in the price of the underlying asset. Mathematically, it is the first derivative of the option price with respect to the underlying asset's price:

\[
\Delta = \frac{\partial C}{\partial S} = \Phi(d_1)
\]

In delta hedging, the goal is to construct a portfolio that is delta-neutral, meaning that the portfolio's value is insensitive to small changes in the underlying asset's price. This is achieved by holding a position in the underlying asset that offsets the delta of the option position. For a long call option position, the delta is positive, so the hedging strategy involves shorting \( \Delta \) units of the underlying asset.

### Gamma, Theta, and Other Greeks

Beyond delta, other Greeks play significant roles in the behavior of options and the effectiveness of hedging strategies:

- **Gamma (\( \Gamma \))**: Measures the rate of change of delta with respect to the underlying asset's price. A long option position has positive gamma, implying that delta increases as the underlying price increases and vice versa. High gamma necessitates more frequent rebalancing to maintain a delta-neutral portfolio.
  
- **Theta (\( \Theta \))**: Represents the sensitivity of the option's price to the passage of time, also known as time decay. For long options, theta is negative, meaning the option loses value as time progresses, all else being equal.

- **Vega (\( \nu \))**: Measures the sensitivity of the option's price to changes in volatility.

Understanding these Greeks is crucial for managing the risks associated with options trading and implementing effective hedging strategies.

### Transaction Costs and Practical Considerations

In an idealized, frictionless market, continuous delta hedging would perfectly replicate the option's payoff at expiration. However, in reality, several factors lead to deviations from this ideal:

- **Transaction Costs**: Each rebalancing of the portfolio incurs costs such as bid-ask spreads, commissions, and slippage. Frequent rebalancing, especially in high-gamma scenarios, can accumulate significant costs over time.

- **Discrete Rebalancing**: Continuous rebalancing is impossible in practice. Rebalancing at discrete intervals (e.g., daily, weekly) introduces additional risks and costs, as the portfolio's delta can drift between rebalancing times.

- **Volatility Estimation**: Accurate estimation of volatility is challenging. Misestimation can lead to incorrect delta calculations, affecting the hedging effectiveness.

- **Model Assumptions**: The Black-Scholes model assumes constant volatility and interest rates, which may not hold in dynamic market conditions.

These factors contribute to the negative expected value of a delta-hedged long option position when implemented realistically, as seen in the simulation results.

---

## Implementation

### Data Collection and Preprocessing

The simulation relies on historical data for the underlying asset, risk-free rates, and volatility proxies (e.g., VIX). Data is fetched using the `yfinance` library and processed as follows:

- **Stock Data**: Historical closing prices are used to compute log returns.
  
- **Risk-Free Rate**: Obtained from a relevant financial instrument (e.g., 5-Year Treasury Rate) and annualized.
  
- **Volatility Proxy**: VIX or other implied volatility measures are used as proxies for the underlying asset's volatility.

### Volatility Estimation

Volatility is estimated using two approaches:

1. **GARCH Model**: The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model captures time-varying volatility by modeling the conditional variance of returns. This provides a smoothed estimate of annualized volatility used in the hedging strategy.

2. **VIX Proxy**: Implied volatility from market instruments like the VIX index serves as an alternative volatility estimate.

### Delta Hedging Strategy

The core components of the delta hedging simulation include:

- **Initial Positioning**: Establishing a portfolio by purchasing a call option and delta-hedging by shorting the underlying asset based on the option's delta.

- **Simulation of Trades**: Iteratively rebalancing the portfolio at specified intervals (`day_rebalancing`) and rolling the option position (`day_rolling`) to maintain the hedge. This process accounts for transaction costs and time decay.

- **Parameter Sweep**: Testing various combinations of rebalancing (`day_stock`) and rolling (`day_option`) frequencies to assess their impact on the portfolio's final value.

### Parameter Sweep and Analysis

To evaluate the strategy's robustness, the simulation iterates over different values of `day_stock` (rebalancing frequency) and `day_option` (rolling frequency). For each combination, the final portfolio value is recorded, and the results are visualized using heatmaps to identify optimal rebalancing and rolling schedules.

---

## Usage

### Prerequisites

Ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `arch`
- `tqdm`

You can install them using `pip`:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance arch tqdm
```

### Running the Simulation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/MMA707.git
   cd MMA707
   ```

2. **Configure Parameters:**

   Adjust the parameters in the main script as needed, such as the underlying symbol, risk-free rate proxy, volatility proxy, and historical data period.

3. **Execute the Script:**

   Run the main script to perform the delta hedging simulation and parameter sweep.

   ```bash
   python delta_hedging_simulation.py
   ```

4. **View Results:**

   The simulation will generate CSV files containing the portfolio data and produce heatmaps illustrating the final portfolio values across different parameter combinations.

### Interpreting the Results

- **Final Portfolio Value:** Indicates the profitability of the delta hedging strategy after accounting for transaction costs and other factors.

- **Heatmaps:** Visual representations showing how different rebalancing and rolling frequencies affect the strategy's performance. Optimal parameters are those that maximize the final portfolio value while minimizing costs.

- **Cumulative Fees and Delta Drift:** Provide insights into the costs incurred and the effectiveness of maintaining a delta-neutral portfolio over time.

---

## Results

*Note: Include sample results or summaries here if available.*

---

## Conclusion

The delta hedging strategy implemented in this repository demonstrates the practical challenges of replicating option payoffs in real-world markets. While the theoretical foundation provided by the Black-Scholes model suggests that continuous hedging can perfectly replicate option payoffs, practical considerations such as transaction costs, discrete rebalancing, and volatility estimation errors lead to a gradual loss of portfolio value over time. The parameter sweep analysis highlights the trade-offs between hedging frequency and transaction costs, underscoring the importance of optimizing rebalancing schedules to mitigate losses.

---

## References

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
- Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics and Management Science, 4(1), 141-183.
- Leland, H. E. (1985). *Option Pricing and Replication with Transactions Costs*. Journal of Finance, 40(3), 667-685.
- Boyle, P., & Vorst, T. (1992). *Option Replication in Discrete Time with Transaction Costs*. Journal of Finance, 47(5), 1731-1751.
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [ARCH Package Documentation](https://arch.readthedocs.io/en/latest/)
