# Research of Limit order book behavior about fill rate
## Fill rate prediction
### Visualize fill rate
**DEFINE**: the fill rate here is the fill prob. estimated in a certain time and price grid\
To make it more simple we'll focus mainly on the large tick assets to aviod the decision of size price grid\
**To do**:
1. fill rate estimation: \
**potential method**:

    - for the time grid is large enough define it by statics of fill events (How many fill events happen in a time grid)
    - evalute it by 'intensity' (number of fill order divided by average time difference between two fill events)
    - integral method (smooth the fill rate change by time function, approximate it)

2. Visualization: \
build the visualization tool for the better observing to its relationship with other market attribute\
    **Must have function**:
    - `grid_estimator`: quickly estimating the behavior inside one grid (statistical value approximation, examples are those to find fill rate)
    - `env_estimator`: find the macroscopic view of order book at that time grid (Accross different price grid, e.g. order book imbalance)
    - `behavior_estimator`: able to divide the time grid more and look for uniform behavior of orders (price match mechanism detection,i.e. cancel the order and add it with new price/size uniformly, like the uniform price close to best prices)\
        **NOTE**: This one might be the hardest, it can be optional if spending too much time on it
3. Latent finding \
Based on the obsevation, try to do the modeling \
**Modeling reference**:

    - [Avellaneda Stoikov model](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)


## Market Simulation Building
Markovian market simulated environment for model validation/testing\
Base: [Bridging the Reality Gap in Limit Order Book Simulation](https://arxiv.org/pdf/2603.24137)\
**To Do**:

    - build exactly as the paper 
    - by appreciation to the market data, try to decrease the complexity of model or find better factors (latents finding)
    - verification with the real market data
    - Extreme testing the model (reduce the factor and see how it perform, does it deviate a lot?)

## Strategy based on fill rate
