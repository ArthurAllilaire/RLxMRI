Contents:

# Vanilla Policy Gradient Optimisation

delta log theta

# TRPO

clip advantage function
works for continuous spaces.
Blog post:

https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

# Optimisations specific to my usecase

On policy exploration is the bottle neck thanks to simulator time costs - this can be parallelised and approximated to increase training speed.

Change your advantage function by bumping up probability for actions with high expected return  but keep KL divergence within good bounds instead of just hoping small changes in theta wont cause massive changes.

