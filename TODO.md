# Things to understand:

1. Gradient spoiling
2. pixel_grid_overlay.jl
- blocks.jl - the new gre_2d_sequence
- test_simulation.jl - why is it there? what does it actually test?
- understand e two code
- send email to andreas plus wayne
- 

check why this is so slow:

Turns out its 5 seconds per simulation

run cr optimal alpha benchlines
merge cr optimal alpha and cr optimal together once understood
understand cr optimal - at least high level

seems like TR is assumed to be 0.5 seconds? turns out thats just hardcoded lowest TR val of E2

# Next steps:
3. The alpha angle - currently fixed at pi/2 but brusters angle is best SNR for given TR and TI - between 0 - pi/2 and usually closer to 0.