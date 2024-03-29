# sentiment-sleuth

Assess the market trend and return coded values indicating the market condition.

## Overview

The tool uses the principles of market analysis to identify trends by analyzing price movements over time. It calculates weighted logarithmic returns on market data (focusing on closing prices) and applies linear regression to assess the strength and direction of market trends. The application is capable of interacting with trading platforms to fetch real-time data for analysis.

Based on the principles of the efficient market hypothesis, which suggests that financial markets fully reflect all available information, I employ weighted logarithmic returns to assess the presence of significant market trends. The weighting system is designed so that individuals earlier in the transaction chain, who possess greater knowledge and access to information, have their prices given increased importance, reflecting their higher informational value.

## Features

-   **Weighted Log Returns**: Applies weights to logarithmic returns, emphasizing earlier data points in the series, which are presumed to carry more informational value.
-   **Market Condition Assessment**: Categorizes the market into various states such as flat, trending up (strongly or weakly), and trending down (strongly or weakly) based on volatility and trend strength.
-   **Data Fetching and Authentication**: Includes the `IGAPIClient` class to authenticate and fetch real-time market data from trading platforms.
-   **Visualization**: Provides visual representation of log returns with their respective weights for easier analysis and understanding.
-   **Customizable Analysis**: Users can adjust parameters like the volatility threshold to suit different analysis needs.
  
<img width="551" alt="image" src="https://github.com/tg12/sentiment-sleuth/assets/12201893/f8dafc50-8040-413c-90cd-b50b04888ac9">

## Contributing

Contributions to improve the tool or extend its functionalities are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is provided "AS IS", without warranty of any kind. By using this tool, you acknowledge the inherent risks associated with financial market analysis.

## Credits and Acknowledgments

If you find this Market Trend Analysis Tool useful in your research, projects, or in any other capacity, I kindly ask for acknowledgment by crediting me as the original author. A simple reference to my contribution can help support the development of more open-source tools and resources in the financial analysis domain.
