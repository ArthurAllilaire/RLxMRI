Contents:

1. Bug #1
2. Bug #2


# The debugging process:


# Residual per-shot signal drift past ~270s
Same FP collapse as the first bug, just in two different sites. Issue raised: https://github.com/JuliaHealth/KomaMRI.jl/issues/788 

PR: https://github.com/JuliaHealth/KomaMRI.jl/pull/789 


### Known issue: floating-point accumulation in KomaMRI

During development two floating-point accumulation bugs were discovered in `KomaMRIBase.jl`. The first caused per-shot signal drift in long sequences (> ~270 s total duration) when the gradient moment accumulator overflowed. The second was the same root cause at a different accumulation site. Both were diagnosed, reported (GitHub issue #788), and fixed in a fork (`ArthurAllilaire/KomaMRI.jl`, branch `fix/grad-fp`, PR #789). The library currently depends on this fork at a pinned commit; the fix is awaiting upstream merge. This dependency is the only deviation from the registered Julia package registry.