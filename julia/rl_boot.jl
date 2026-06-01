# Research-layer RL environments for RLxMRI.
# Included into Main after `using MRISystemPhantom` so all library symbols
# are in scope. Defines E1Env, E2Env and their step/reset functions.
using Random, Statistics, LinearAlgebra
using Suppressor
include(joinpath(@__DIR__, "rl", "e1.jl"))
include(joinpath(@__DIR__, "rl", "e2.jl"))
