__precompile__(true)
module ePPR

include("algorithm.jl")

# export all symbols
for n in names(@__MODULE__, all=true)
    if Base.isidentifier(n) && n ∉ (nameof(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

function __init__()
    R"library('MASS')"
    pyplot()
end

end
