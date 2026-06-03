Contents:

- Framing this as an RL problem


# Objective
Minimise MAPE 


## Action Space

Experimented with quite a few action spaces:

- Fix/free TE
- Fix/free alpha
- Log grid / continuous space for TI
- 

A lot of this experimentation was done on the buggy simulator and so cannot be trusted fully.

## Observation space

- Do we use sigma - the uncertainity of the fitter?
- How do we calculate that uncertainity?
- Do we pass in the raw pixels - why would that be helpful?

## Fitter, cleaning, 


## Julia vs. Python packages used

- **Julia 1.11 vs 1.12 split**: juliacall requires Julia ≤ 1.11. The Python-facing runtime at `python/julia_runtime/` pins 1.11; the main project can use 1.12. Do not break this separation.

## Phantom setup

- Single z-slice including water and the T1 plates
- no rotation or translation necessary since I only ever passed in the T1 estimates and not image signal to the RL agent.

## Assumptions

- Transient Mxy is a negligble error - later added gradient spoiling to get rid of this.