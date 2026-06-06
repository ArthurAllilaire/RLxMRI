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

## What needs to be redone...

Sections:

1. fp_bugs.md - need to regenerate the plots. Then regenerate the writing
How come nobody found this before? usually people use gradients not hard RF pulses - when not 2d

2. k-space explanation - need to add it 

3. Take a photo of the 3mm phantom T1 voxelised src/asset and add that

Code that needs to be re-run:

1. Multi-fidelity runs - all the plots haven't been saved...
2. 