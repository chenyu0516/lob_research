# Progress Report 2026 April
## Research about limit order book behavior
Paper I read:
1. [Simulating and analyzing order book data: The queue-reactive model](https://arxiv.org/abs/1312.0563)
2. [Bridging the Reality Gap in Limit Order Book Simulation](https://arxiv.org/pdf/2603.24137)
3. [Limit Order Book Dynamics in Matching Markets:
Microstructure, Spread, and Execution Slippage](https://arxiv.org/abs/2511.20606)

I'm trying to find the "pattern in the limit order book data. Instead of diving in like no-head fly, I research over their works and figure out my own version:

In the first and the third paper, they all mensioned the "intension" of filling a order. So I model the behavior of limit order book as a decision process to fill! I'll discuss it through the whole lifetime of a limit order. First of all, a limit order is added to the book, the intension of this order to be filled depends on how far the order is placed from the best price and how large the size of the order is. The more easy the order to be filled the higher intension the order-placing agent has. Based on the intension, the order might be changed/cancelled due to the move of the best price. As the order's price getting closer to best price, the probability that order will be filled is going higher. The order is changed/cancelled when the trading alogrithm detects the approaching of best price. (might be the approaching rate, order book imbalance or distance smaller a threshold, this will need empirical validation)

In conclusion, I try to build my own appreciation of the market data started from the other's opnion. But the empirical study is needed.

## Tools building for empirical study
If you see the root directory of this github repo, you will see it is a huge project for limit order book visualization. I've finished the data processed section in early April. In [README.md](../README.md), there are technical details that show how I process the data from coinbase and databento. The visualization part haven't been done yet. Professor you might notice that I keep asking about how to look at the limit order book data lately. I found out that limit order book data is such a high dimension data that can't be understand by plotting them out just like volume hunting trader. Professor's opnion is that I need to find a simple idea behind it, so I go for literature research first. For the dataprocess part, now this github repo can take the data from coinbase/databento, do the data process(make them to the unified format) and save as .parquet file, which is much more faster to access than traditional .csv file.

For more detail see [README.md](../README.md)

## Future work
1. empirical validation of my ideas. deadline: 5/5
    * I would like to start my discuss with specific price-time grid ($time\in[t+\Delta t], price\in[p+\Delta p]$), compute the fill rate there
    * find the relation between fill rate and other feature in the same time grid
    * find the relation between fill rate and features in neighbor grid
    * find the relation between fill rate and other feature in the opposite side
    * find the relation between fill rate whole book's behavior

2. mathematical model building for intension filling limit order. 
Whole theoretical idea should be confirmed before 5/7 and built in to decision process before 5/14
3. optimal execution alogrithm? (I'm not sure about it)

## Questions:
* I'm curious about if the fill rate estimation can be done by doing statistics or other analyze method over several types of optimal execution algorithm. If they are "optimized" they might mostly have the same behavior at the same moment. I'ld like to know if it is admissible.