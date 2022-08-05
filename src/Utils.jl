using Graphs

include("./Types.jl")

function Graphs.Graph(history::History)
    max_id = maximum(vcat(collect.(history)...))
    g = SimpleGraph(max_id)
    for event in history
        add_edge!(g, event)
    end
    return g
end

function rem_alone!(g::Graph)
    alones = findall(d -> d == 0, degree(g))
    rem_vertices!(g, alones)
    return g
end

# thanks: https://discourse.julialang.org/t/split-vector-into-n-potentially-unequal-length-subvectors/73548/3
function makechunks(X::AbstractVector{T}, n::Int) where {T}
    L = length(X)
    c = L ÷ n
    Y = Vector{Vector{T}}(undef, n)
    idx = 1
    for i in 1:(n - 1)
        Y[i] = X[idx:(idx + c - 1)]
        idx += c
    end
    Y[end] = X[idx:end]
    return Y
end

function get_birthstep(aid, history)
    return findfirst(x -> x[1] == aid || x[2] == aid, history)
end

"""
    get_birthsteps(history::History)::Dict{Int, Int}

history中に存在するエージェントの誕生ステップをDictで返す
"""
function get_birthsteps(history::History)::Dict{Int,Int}
    src = history .|> first
    dst = history .|> last
    t = 1:length(src) |> collect
    df = DataFrame(; t, src, dst)

    dfsrc = rename(select(unique(df, :src), [:t, :src]), :src => :aid)
    dfdst = rename(select(unique(df, :dst), [:t, :dst]), :dst => :aid)

    replacemissing! = df -> begin
        for col in eachcol(df)
            replace!(col, missing => length(history) + 1)
        end
        return df
    end
    joined = outerjoin(dfsrc, dfdst; on=:aid, renamecols="_s" => "_d") |> replacemissing!
    joined.t = min.(joined.t_s, joined.t_d)

    return Pair.(joined.aid, joined.t) |> Dict
end

function parseint(x)::Int
    return parse(Int, x)
end

function convertint(x)::Int
    return convert(Int, x)
end

function remzero(ar::Array)
    return filter(v -> v != 0, ar)
end
