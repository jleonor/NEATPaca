# NEATPaca: An Evolutionary Approach to Learning How to Trade
![ay_lmao](https://i.postimg.cc/RhqTkFSh/NEATPaca.webp)
## Introduction
In this project, I use the NEAT (Neuro Evolution of Augmenting Topologies) algorithm within a Python-based trading platform to develop an approach to algorithmic trading. The core of the project revolves around an autonomous trading agent capable of buying and selling positions across various currencies, with each action recorded and analyzed to enhance future decision-making. Leveraging the comprehensive data provided by the Alpaca API, the agent utilizes an array of technical indicators across multiple timeframes and window sizes, each normalized to facilitate AI interpretation. This enables the agent to not only execute trades based on sophisticated pattern recognition but also evolve its strategies over time through the NEAT algorithm's evolutionary processes.

## Alpaca API
In the project, leveraging the Alpaca API has been instrumental in streamlining the data acquisition and trading execution processes. Alpaca, a platform known for its commission-free trading, offers a robust API that enables developers like myself to access real-time market data, execute trades, and manage portfolios programmatically. This integration is vital for the project's success as it ensures timely and efficient market interactions. By utilizing Alpaca, I can focus on refining the trading algorithm and strategies, rather than the intricacies of data management and trade execution.

### Benefits of Using the Alpaca API
- **Real-time Data Access:** The API provides minute-by-minute trading data, which is crucial for the project's need to analyse market trends and make informed decisions quickly.
- **Commission-Free Trading:** This feature allows for a more extensive range of trading experiments without the burden of transaction costs, making the project more viable and cost-effective.
- **Paper Trading Environment:** Alpaca's paper trading option is an invaluable tool for testing trading strategies in a risk-free environment. It simulates real market conditions, enabling me to refine strategies before going live.
- **Seamless Transition to Live Trading:** Once a strategy is proven successful in paper trading, the API allows for an easy switch to live trading. This flexibility supports the project's progression from theoretical modelling to practical application.

## Technical Indicators/Features for AI Training
In the development of my algorithmic trading platform, the integration and analysis of technical indicators play a pivotal role in training the AI to make informed trading decisions.

### Understanding AI Training with Technical Indicators
At the core of the AI's training is its exposure to a wide array of technical indicators across various timeframes and window sizes. This exposure allows the AI to analyse market conditions from multiple perspectives. For example, the Relative Strength Index (RSI) might be viewed in 7-day and 14-day windows, and similar flexibility applies to the timeframes of data it analyses, such as 5 minute, 30-minute, and 1-hour intervals. This multi-faceted approach enables the AI to develop a comprehensive understanding of market dynamics. However, it's crucial to recognize that increasing the number and complexity of these windows and timeframes directly impacts the training duration.

#### Key Technical Indicators Employed
- **Moving Averages (MA)**
- **Bollinger Bands**
- **Stochastic Oscillator**
- **Moving Average Convergence Divergence (MACD)**
- **On-Balance Volume (OBV)**
- **Price Percent Change**
- **Parabolic SAR**
- **Average Directional Index (ADX)**
- **Relative Strength Index (RSI)**
- **Commodity Channel Index (CCI)**
- **Money Flow Index (MFI)**
- **Aroon Oscillator (AROONOSC)**
- **Chande Momentum Oscillator (CMO)**
- **Long-Term Oscillation (LTOCS)**

These indicators measure various aspects of market conditions, such as momentum, volume, volatility, and trend strength, providing the AI with a rich dataset for analysis.

#### The Importance of Normalization
Normalization of data is a critical step in the training process. It converts the raw values from technical indicators into a standardized format, usually within a 0 to 1 range. This process is essential for two reasons. First, it helps the AI to uniformly interpret and compare data across different indicators and securities. Second, it prevents the algorithm from being overwhelmed by large numerical values, ensuring that no single indicator disproportionately influences the decision-making process. In practical terms, I often translate the indicators into binary interpretations to simplify the AI's analysis. For example, if a moving average crosses below the close price, this event is represented as a 1; if it crosses above, it's represented as a 0. This binary representation streamlines the AI's ability to recognize patterns and make trading decisions based on clear, actionable signals. Alternatively, the RSI is always between 100 and 0, which can easily be converted to values between 1 and 0.

## NEAT Algorithm
The Neuro Evolution of Augmenting Topologies (NEAT) algorithm is a cutting-edge approach to machine learning that mimics the process of natural evolution to develop complex models. At the heart of NEAT is the concept of evolving a population of neural networks through a process similar to natural selection, where only the fittest individuals pass their characteristics to the next generation.

In the context of algorithmic trading, I utilize NEAT to evolve a population of trading agents. These agents start with basic trading strategies and, over time, evolve to adapt to the complexities of the market. The evolution process involves mutation and crossover of the agents' neural networks, enabling the development of more sophisticated strategies through the combination of successful traits from previous generations.

### Benefits in Trading
- **Adaptability:** NEAT's evolutionary approach allows trading strategies to continuously improve and adapt to changing market conditions without manual intervention.
- **Innovation:** By exploring a wide range of strategy combinations, NEAT can uncover unique and effective trading strategies that might not be immediately obvious to human traders.

### Negatives
- **Computational Intensity:** The process of evolving and testing multiple generations of trading agents can be computationally demanding, requiring significant resources and time.
- **Risk of Overfitting:** There's a potential for the algorithm to become too finely tuned to historical market data, which might not accurately predict future conditions, leading to suboptimal trading performance.
- **Lack of Instant Adaptability:** While NEAT excels at developing strategies over multiple generations, it may not respond instantaneously to sudden, dramatic market shifts, as its learning process is based on the accumulation of evolutionary progress over time.

## NEAT Trading Agent
The NEAT trading agent plays a pivotal role in the project. This agent employs the NEAT algorithm to autonomously navigate the stock or currency markets, with its functionality characterized by both its innovative capabilities and certain limitations.

### Capabilities
- **Decision-Making:** The agent uses a neural network, evolved through the NEAT algorithm, to analyse a specific set of inputs for making trading decisions. These inputs include technical indicators, the potential gain of a trade, whether the agent currently holds an open position, and the duration for which a position has been held. This focused analysis allows the agent to identify trading opportunities that align with its evolutionary learning.
- **Evolutionary Learning:** Leveraging the NEAT algorithm, the agent evolves its trading strategies over time. Each trade and its outcome are meticulously recorded, contributing to the agent's learning process. Successful strategies are retained and refined across generations, enhancing the agent’s ability to make profitable decisions.
- **Trade Execution:** The agent is programmed to execute trades decisively: it invests all available funds when buying and sells all held assets when selling. This all-in approach is aimed at capitalizing on identified opportunities, though it inherently increases risk exposure.
- **Performance Tracking:** A comprehensive log of trading activities is maintained, enabling the evaluation of the agent's performance over time. This tracking aids in assessing the effectiveness of the agent's strategies under various market conditions.

### Limitations
- **Risk Management:** The agent does not incorporate specific risk management strategies, focusing solely on the potential gains without mechanisms for limiting losses or protecting profits. This could lead to heightened risk exposure.
- **Position Diversification:** Limited to holding a single position per currency at any given time, the agent's ability to diversify its investment and mitigate risk is constrained.
- **Adaptation to Market Changes:** Although capable of evolving based on historical data and past outcomes, the agent may struggle with rapid adaptation to new or unexpected market conditions, affecting its real-time response to volatility.
- **Balancing challenge:** The evolution of the agent’s decision-making capabilities and strategies necessitates a careful balance between computational resources and the pursuit of improved trading outcomes. Additionally, there's a paramount challenge in preventing overfitting, ensuring that the agent remains versatile and effective in dynamic market environments.
