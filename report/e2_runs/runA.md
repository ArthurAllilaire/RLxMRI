Contents:

- reward functions
 - initially MAPE reward per policy
 - changed to delta MAPE and terminal only
 - ran smoke tests with analytical 30k steps
 - quickest convergence was delta log MAPE so used that for the first training run

- Plotting the Pareto Curve
 - My next challenge!#
 

# Choosing reward functions

What makes a good reward function?

- Close to the goal
- Hard to game
- Informative + attributabal

The goal of this agent is to beat the benchmarks on MAPE. 

Initial idea: reward at every timestep = MAPE. Where one timestep is a block