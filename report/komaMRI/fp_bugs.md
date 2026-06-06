Prompt:

I need you to draft the KomaMRI chapter for my report - this should be relatively short - high level explanation of the simulator followed by how i found the bugs explanation of that bit of the simulator and how i fixed them and pushed PR. add it to fp_bugs_final.md file - I will then convert to latex and add to report. use the different figures below - caption them and add extra details in comments as you go through your investigation - the different scripts usually have headers as to what i was thinking. Also look at the scripts/koma_investigations folder for the investigations you did - i understand these less but if relevant add. make sure you can read the issues and PRs - if not i can paste it in. If the investigation looked at tranverse spoiling and there is a script with some numbers of how much that does play a role please add an indepth section of that in this - i will relocate it to my evaluation section. Please try and gather a rich context before writing and cutting/editing, leave only the most insightful but keep a good narrative arc and leave comments with bits you find and cut in case i would like to include later. 

I have already written this section but have since lost it so have addded my rough notes on the overall flow / what i can remember - bring it back to report ready! I have also lost the run folder - but if there are interesting scripts you would like to re-run to see what the output was - use the branch that can be found at (../../../fp_bugs_RLxMRI) and run them in there. see the CLAUDE.md in there for how to do that.

The overall flow should be:

# Koma MRI
KomaMRI is the simulation environment used throughout. found a bug in the edge-boundaries that made them collapse - this chapter explains that - link back to challenge/achievement - and also describes how the bug was found

## Initial explanation of KomaMRI

KomaMRI is built around the simulate call - a user calls:

raw = simulate(phantom, sequence, scanner; sim_params)


The explanation you give for koma should have the same starting sentence as the Koma docs see bottom: 

KomaMRI simulates the magnetization of each spin of a Phantom for variable magnetic fields given by a Sequence. 

Make it concrete a Phantom is a cloud of spins - we get our Phantoms for this project from MRISystemPhantom.jl. Each spin isochromat is a group of nearby nuclei assumed to have a single position (x,y,z), tissue parameters (T1,T2,T2*,RHO, DELTA W) and resonance frequency. magnetisation vector

KomaMRI then splits sequence onto a discretised time grid by delta t_rf and delta t and uses that to simulate the bloch equations - then each adc reading it sums up over all spins for each t. splits based on RF active - then have to model full magnetisation vector rotations otherwise easier relaxation curves so saves memory/efficient. This is why we chose it along with CPU threads below.

Assumption that eacah spin is independent - this is standard for MRI simulators. lets you parallelise across CPU threads and GPU kernels - just sum up at each ADC.

Then go into other assumptions of simulator - model bulk T2* instead of microscopic spin to spin interaction. Hardware limits not taken into account - eddy currents, B0/1 inhomegenity, off resonance etc. and overall these are fine for our experiment. others that you can think of




## Initial symptom: noisy recovery curves

Simple experiment no-noise inversion recovery T1 fit. fit the the monoexponential recovery curve by smapling points by varying TI - include the M_z(TI) = equation. See per-shot variation that means the simple T1 recovery curve cannot fit. So the simulator, sequence or reconstruction/fit pipeline must be wrong - or there is some physical phenomena I haven't taken into account.

Investigated and ruled out the MRI failure modes.

Transverse spoil assumption - could have leftover magnetisation via RF pulse refocused and into ADC reading. tried a gradient spoiler added at TR and around refocus (i think double check) increased the error? see below for the figs

Also increasing TR should be a solution allow relaxation so Mxy dies and Mz fully recovered - no shot drift. made error worse. in hindsight as longer simulation time which is what triggers the bug

Then thought Gibbs ringing - hence the hamming window support in MRISystemPhantom and ROI - they didn't solve it either - also tried phase sensitive not magnitude and maybe we're assuming inital phase of 0 maybe we can't assume that or it changes - thought about a lot see the different scripts. 

Contents:

1. Bug #1
2. Bug #2


BELOW YOU HAVE EVEN ROUGHER NOTES THAT ARE me remembering what i did.

# The debugging process:

T1 fit with no noise - find the recovery curves - why are they so noisy? look at k-space. At some point acquisition time becomes meaningless. Turns out its a fp issue - the boundary used for interpolation of the RF pulses means that an extra delta t is getting RF/2 which leads to quite an issue.

Couldn't get fit down to below 39%??

![](figs/eae656a_current-koma/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)
produced by t1_fit_vs_true


![](figs/eae656a_current-koma/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves_koma.png)

Could it be the tranverse spoil assumption that is the issue? Bumped TR - didn't get rid of it made it worse! what is going on (obviously because time was bigger so more buggy pulses) and tried adding a spoil to the pulse:

See this folder for spoil runs: scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise_spoil

Better but still 20% error? and why are those points not on the curve! This can't just be numerical imperfections there must be a physical reason.


Initially thought it was Gibbs ringing or voxelisation error - so plotted the output of a signal:

pixel_overlay
This is where it finally cracked...
![](figs/eae656a_current-koma/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)
acquisition along y axis is time - we were doing so well and then....
and also that is not gibbs ringing - that is way too much streaking.


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

![](figs/7ceced7_fixed/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)

![](figs/7ceced7_fixed/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)

![](figs/7ceced7_fixed/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves_koma.png)

Thats better!


## Koma docs on simulate:



General Overview
KomaMRI simulates the magnetization of each spin of a Phantom for variable magnetic fields given by a Sequence. It is assumed that a single spin is independent of the state of the other spins in the system (a key feature that enables parallelization). Furthermore, there are defined two regimes in the Sequence: excitation and precession. During the latter, the excitation fields are nulled and are useful for simplifying some physical equations.

The are more internal considerations in the KomaMRI implementation. The Figure 1 summarizes the functions called to perform the simulation.


Figure 1: The sequence seq is discretized after calculating the required time points in the wrapper function simulate. The time points are then divided into simulation blocks to reduce the amount of memory used. The phantom obj is divided into Nthreads, and KomaMRI will use either run_spin_excitation! or run_spin_precession! depending on the regime. If an ADC object is present, the simulator will add the signal contributions of each thread to construct the acquired signal sig[t]. All the parameters: Nthreads, max_block_length, max_rf_block_length, Δt_rf, and Δt, are passed through a dictionary called sim_params as an optional parameter of the simulate function.

From the programming perspective, it is needed to call the simulate function with the sim_params dictionary keyword argument. A user can change the values of the following keys:

Parameter	Description
"return_type"	defines the output of the simulate function. Possible values are "raw", "mat", and "state", corresponding to outputting a MRIReco RawAcquisitionData, the signal values, and the last magnetization state of the simulation, respectively.
"sim_method"	defines the type of simulation. The default value is Bloch(). Other built-in methods include BlochDict(), BlochMagnus1(), BlochMagnus2(), and BlochMagnus4(). You can also define custom methods without altering the KomaMRI source code; for details, see Simulation Method Extensibility.
"Δt"	raster time for gradients.
"Δt_rf"	raster time for RFs.
"precision"	defines the floating-point simulation precision. You can choose between "f32" and "f64" to use Float32 and Float64 primitive types, respectively. It's important to note that, especially for GPU operations, using "f32" is generally much faster.
"max_block_length"	maximum number of time steps per precession block.
"max_rf_block_length"	maximum number of time steps per RF excitation block.
"Nthreads"	divides the Phantom into a specified number of threads. Because spins are modeled independently of each other, KomaMRI can solve simulations in parallel threads, speeding up the execution time.
"gpu"	is a boolean that determines whether to use GPU or CPU hardware resources, as long as they are available on the host computer.
"gpu_device"	sets the index ID of the available GPU in the host computer.
For instance, if you want to perform a simulation on the CPU with float64 precision using the BlochDict() method (assuming you have already defined obj, seq and sys), you can do so like this:


# Set non-default simulation parameters and run simulation
sim_params = KomaMRICore.default_sim_params() 
sim_params["gpu"] = false
sim_params["precision"] = "f64"
sim_params["sim_method"] = BlochDict()
raw = simulate(obj, seq, sys; sim_params)
Additionally, the user must be aware of the functions run_spin_excitation! and run_spin_precession! which defines the algorithm for excitation and precession regimes respectively and can be changed by the user without modifying the source code (more details at Simulation Method Extensibility).

Previous simulation, the Sequence is discretized to consider specific time points which are critical for simulation. The user can control the time between intermediate gradient samples with the parameter Δt. Similarly, the parameter Δt_rf manages the time between RF samples, and can be relatively large for 2D imaging where the slice profile is less relevant.

For RF excitation with rapidly changing fields, the BlochMagnus1(), BlochMagnus2(), and BlochMagnus4() methods can improve accuracy at the same Δt_rf, or allow a larger Δt_rf for faster simulations. See Magnus Bloch Methods.

Computation Efficiency
To reduce the memory usage of our simulator, we subdivide time into simulation blocks. KomaMRI classifies each block in either the excitation regime or the precession regime before the simulation.

We increased the simulation speed by separating the calculations into Nthreads and then performing the GPU parallel operations with CUDA.jl . This separation is possible as all magnetization vectors are independent of one another.

Simulation Method Extensibility
In Julia, functions use different methods based on the input types via multiple dispatch. We used this to specialize the simulation functions for a given sim_method <:SimulationMethod specified in sim_params. For a given simulation method, the function initialize_spin_state outputs a variable Xt <: SpinStateRepresentation that is passed through the simulation (Figure 1). For the default simulation method Bloch, the spin state is of type Mag, but can be extended to a custom representation, like for example EPGs44 or others. Then, the functions run_spin_excitation! and run_spin_precession! can be described externally for custom types sim_method and Xt, extending Koma’s functionalities without the need of modifying the source code and taking advantage of all of Koma’s features.

Bloch Simulation Method
This is the default simulation method used by KomaMRI, however it can always be specified by setting the sim_method = Bloch() entry of the sim_params dictionary. In the following subsection, we will explain the physical and mathematical background and some considerations and assumptions that enables to speed up the simulation.

Physical and Mathematical Background
The Bloch method of KomaMRI simulates the magnetization of each spin by solving the Bloch equations in the rotating frame:

with  the gyromagnetic ratio,     the magnetization vector, and

the effective magnetic field.  is the proton density,  and  are the relaxation times, and  is the off-resonance, for each position.