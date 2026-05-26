struct SphereDescriptor
    centre::NTuple{3,Float64}   # m
    radius::Float64             # m
    ρ::Float64
    T1::Float64                 # s
    T2::Float64                 # s
    T2s::Float64                # s
    delta_w::Float64            # rad/s
    label::Symbol
end
