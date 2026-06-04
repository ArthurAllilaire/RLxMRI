Prompt:

I need you to draft the KomaMRI chapter for my report - this should be relatively short - high level explanation of the simulator followed by how i found the bugs explanation of that bit of the simulator and how i fixed them and pushed PR. just update this .md file - I will then convert to latex and add to report. use the different figures below - caption them and add extra details in comments as you go through your investigation - the different scripts usually have headers as to what i was thinking. Also look at the scripts/koma_investigations folder for the investigations you did - i understand these less but if relevant add. make sure you can read the issues and PRs - if not i can paste it in. If the investigation looked at tranverse spoiling and there is a script with some numbers of how much that does play a role please add an indepth section of that in this - i will relocate it to my evaluation section. Please try and gather a rich context before writing and cutting/editing, leave only the most insightful but keep a good narrative arc and leave comments with bits you find and cut in case i would like to include later.

Contents:

1. Bug #1
2. Bug #2


# The debugging process:

T1 fit with no noise - find the recovery curves - why are they so noisy? look at k-space. At some point acquisition time becomes meaningless. Turns out its a fp issue - the boundary used for interpolation of the RF pulses means that an extra delta t is getting RF/2 which leads to quite an issue.

Couldn't get fit down to below 35%??

![](/scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)
produced by t1_fit_vs_true

![](/scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves.png)

Could it be the tranverse spoil assumption that is the issue? Bumped TR - didn't get rid of it made it worse! what is going on (obviously because time was bigger so more buggy pulses) and tried adding a spoil to the pulse:

See this folder for spoil runs: scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise_spoil

Better but still 20% error? and why are those points not on the curve! This can't just be numerical imperfections there must be a physical reason.


Initially thought it was Gibbs ringing or voxelisation error - so plotted the output of a signal:

pixel_overlay
This is where it finally cracked...
![](/scripts/runs/with_fp_bugs/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)
acquisition along y axis is time - we were doing so well and then....



Maybe worth mentioning:
- Thought it could be phase sensitive? magnitudes being mis-represented? 
- Assumption about the starting phase phi
- problem with pixel boundaries or interpolation: sampling 64x32 was that right?? could it be on the wrong place? 

- Basically deep dive into every possible physical reason as to why this could be - i really should have plotted some iamges...


Hmm lets run the simplest verification script
see: scripts/koma_bug_minimal.jl - ah we've got an issue

## Explain what we found and how we fixed - look at issues and PRs - explain how Koma works


# Residual per-shot signal drift past ~270s
Same FP collapse as the first bug, just in two different sites. Issue raised: https://github.com/JuliaHealth/KomaMRI.jl/issues/788 

PR: https://github.com/JuliaHealth/KomaMRI.jl/pull/789 


### Known issue: floating-point accumulation in KomaMRI

During development two floating-point accumulation bugs were discovered in `KomaMRIBase.jl`. The first caused per-shot signal drift in long sequences (> ~270 s total duration) when the gradient moment accumulator overflowed. The second was the same root cause at a different accumulation site. Both were diagnosed, reported (GitHub issue #788), and fixed in a fork (`ArthurAllilaire/KomaMRI.jl`, branch `fix/grad-fp`, PR #789). The library currently depends on this fork at a pinned commit; the fix is awaiting upstream merge. This dependency is the only deviation from the registered Julia package registry.

# First one accepted waiting on the second...

## Happy ending!

Better pixel_overlay - ringing should never be that big - but tbf i didn't know that - maybe no spin background vs big spin could cause these issues?

![](/scripts/runs/pixel_grid_overlay/npe32_nfe64_fov0p2_vox1p0mm_TI0p1_spoil/pixel_grid_overlay_variants.png)

![](/scripts/runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/t1_fit_vs_true.png)

![](/scripts/runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/recovery_curves_koma.png)

Thats better!