include("./Types.jl")
include("./Utils.jl")

using DataFrames, Graphs, StatsBase, GLM, LinearAlgebra

function calc_gamma(history::History)
    checkpoints = collect(1:length(history))
    if length(checkpoints) > 500
        checkpoints = range(1, length(history), 500) |> collect .|> floor .|> convertint
    end

    max_id = maximum(vcat(collect.(history)...))

    g = SimpleGraph(max_id)
    data = DataFrame(; X=Int[], Y=Int[])
    for (index, event) in enumerate(history)
        add_edge!(g, event)

        if index ∈ checkpoints
            push!(data, [index, sum(degree(g))])
        end
    end

    ols = lm(@formula(log10(Y) ~ log10(X)), data)

    return coef(ols)[2], data.X, data.Y
end

function calc_cluster_coefficient(history::History)
    global_clustering_coefficient(Graph(history))
end

"""
    calc_connectedness(history::History; offset_start::Bool=true)

ネットワーク成長時にエッジが張られる箇所の傾向の度合いを算出する。

## Returns

```julia
Dict(:NC => Float64, :NO => Float64, :OC => Float64, :OO => Float64)
```
"""
function calc_connectedness(history::History; offset_start::Bool=true)
    calc_start_step = 1
    if offset_start
        calc_start_step = length(history) * 60 / 100
    end

    max_id = max_id = maximum(vcat(collect.(history)...))
    connectedness_cum = [0, 0, 0, 0]
    g = SimpleDiGraph(max_id)
    for (step, event) in enumerate(history)
        src, dst = event

        # 新しいエッジか判定
        is_new = add_edge!(g, src, dst)

        # エッジの追加で三角形が形成されるか判定
        # srcノードとdstノードに共通の隣接ノードがあれば三角形が形成される
        is_open = isempty(common_neighbors(g, src, dst))

        is_old = !is_new
        is_close = !is_open

        if (step >= calc_start_step)

            # 結果の保存
            if is_new && is_close
                connectedness_cum[1] += 1
            elseif is_new && is_open
                connectedness_cum[2] += 1
            elseif is_old && is_close
                connectedness_cum[3] += 1
            elseif is_old && is_open
                connectedness_cum[4] += 1
            end
        end
    end

    return Dict(
        :NC => connectedness_cum[1] / sum(connectedness_cum),
        :NO => connectedness_cum[2] / sum(connectedness_cum),
        :OC => connectedness_cum[3] / sum(connectedness_cum),
        :OO => connectedness_cum[4] / sum(connectedness_cum),
    )
end

"""
    calc_youth_coefficient(history::History, n::Int)

Youth Coefficientを算出する。

## Arguments
- `n::Int` 履歴をいくつに分割して計算するか (defualt: 100)

## Returns
`(youth_coefficient::Float64, empirical_data::Vector{Float64}, predicted_data::Vector{Float64})`

## Reference
_Monechi, B., Ruiz-Serrano, Ã., Tria, F., & Loreto, V. (2017). Waves of novelties in the expansion into the adjacent possible. PLoS ONE, 12(6), e0179303. [https://doi.org/10.1371/journal.pone.0179303](https://doi.org/10.1371/journal.pone.0179303)_
"""
function calc_youth_coefficient(history::History, n::Int=100)
    birthsteps = DynamicNetworkMeasuringTools.get_birthsteps(history)

    # エージェントIDをそれぞれのbirthstepで置き換えたhistory配列
    birth_history = collect(
        zip(
            getindex.(Ref(birthsteps), first.(history)),
            getindex.(Ref(birthsteps), last.(history)),
        ),
    )
    chunks = makechunks(birth_history, n)

    empirical = map(chunk -> mean([first.(chunk); last.(chunk)]), chunks)

    data = DataFrame(; x=1:n, y=empirical)
    ols = lm(@formula(y ~ x), data)
    y = coef(ols)[2] / (length(history) / n)

    predicted = predict(ols)

    return y, empirical, predicted
end

"""
    calc_ginilike_coefficient(history::History)

ジニ係数的な指標を算出する。

# Retruns
`(gini-like_coefficient::Float64, x::Vector{Float64}, y::Vector{Float64})`

# Reference
_Monechi, B., Ruiz-Serrano, Ã., Tria, F., & Loreto, V. (2017). Waves of novelties in the expansion into the adjacent possible. PLoS ONE, 12(6), e0179303. [https://doi.org/10.1371/journal.pone.0179303](https://doi.org/10.1371/journal.pone.0179303)_
"""
function calc_ginilike_coefficient(history::History)
    birthsteps = sort(get_birthsteps(history) |> collect; by=x -> x[2])
    x = (birthsteps .|> (pair -> pair.second) |> competerank) ./ length(birthsteps)
    data = DataFrame(;
        agent=birthsteps .|> x -> x.first, birthstep=birthsteps .|> x -> x.second, x
    )

    cm = countmap(vcat((history .|> collect)...)) |> collect
    _data = DataFrame(;
        agent=cm .|> x -> x.first, f=cm .|> x -> x.second ./ (length(history) * 2)
    )

    data = sort(leftjoin(data, _data; on=:agent), [:birthstep])
    data.y = cumsum(data.f) .|> x -> x > 1.0 ? 1.0 : x

    return (sum(data.y) - sum(data.x)) / sum(data.x), data.x, data.y
end

"""
    most_accessed_agent_birthstep_in_interval(interval::History, birthsteps::Dict{Int,Int})

interaval中でもっともアクセスされたエージェントが誕生したステップ数を返す

## Returns
`(most_accessed_agent_id::Int, birthstep::Int)`
"""
function most_accessed_agent_birthstep_in_interval(
    interval::History, birthsteps::Dict{Int,Int}
)
    elements = vcat(collect.(interval)...)
    most_accessed_agent_id = sort(collect(countmap(elements)); by=x -> x[2], rev=true)[1][1]
    return most_accessed_agent_id, birthsteps[most_accessed_agent_id]
end

"""
    sort_accessed_agent_birthstep_in_interval(interval::History, birthsteps::Dict{Int,Int})

interaval中でもっともアクセスされたエージェントから順番に、誕生したステップ数を返す

## Returns
`(most_accessed_agent_id::Int, birthstep::Int)`
"""
function sort_accessed_agent_birthstep_in_interval(
    interval::History, birthsteps::Dict{Int,Int}
)
    elements = vcat(collect.(interval)...)
    sorted = sort(collect(countmap(elements)); by=x -> x[2], rev=true)
    return sorted .|>
           (aid_access -> aid_access.first) .|>
           (aids -> map(aid -> birthsteps[aid], aids))
end

"""
    calc_recentness(history::History, [tau::Int])

Recentnessを計算する。
"""
function calc_recentness(history::History, tau=1000)
    separators = 1:tau:length(history)
    intervals = [history[separator:min(separator + tau, end)] for separator in separators]

    birthsteps = get_birthsteps(history)
    return (
        [
            most_accessed_agent_birthstep_in_interval(interval, birthsteps) for
            interval in intervals
        ] .|>
        last |>
        sum
    ) / (separators |> sum)
end

"""インターバル内のローカルエントロピーを計算する"""
function __local_entropy(history::History)
    elements = vcat((history .|> collect)...)
    fjs = (countmap(elements) |> values) ./ (length(history) * 2)
    return -(map(fj -> fj * log10(fj), fjs) |> sum) / log10(elements |> unique |> length)
end

"""
    calc_local_entropy(history::History, tau=1000)

ローカルエントロピーを計算する。
"""
function calc_local_entropy(history::History, tau=1000)
    separators = 1:tau:length(history)
    intervals = [history[separator:min(separator + tau, end)] for separator in separators]
    map(__local_entropy, intervals)
end

function calc_heaps(history::History)
    steps = Int[]
    uagents = Int[]
    set = Set{Int}()

    for (index, (src, dst)) in enumerate(history)
        push!(set, src, dst)
        push!(steps, index)
        push!(uagents, length(set))
    end

    x = steps
    y = uagents

    data = DataFrame(; X=log10.(x), Y=log10.(y))
    ols = lm(@formula(Y ~ X), data)

    coef(ols)[2], (x=x, y=y, predicted=10 .^ predict(ols))
end

function calc_zipf(history::History)
    serialized_history = vcat((history .|> collect)...)
    f =
        (
            countmap(serialized_history) |>
            collect |>
            (x -> sort(x; by=y -> y[2], rev=true)) .|>
            (x -> x[2])
        ) / (length(history) * 2)

    x = collect(1:length(f)) / length(f)
    y = f

    ols = lm(@formula(Y ~ X), DataFrame(; X=collect(1:length(f)) .|> log10, Y=f .|> log10))

    return coef(ols)[2], (x=x, y=y, predicted=10 .^ predict(ols))
end

function calc_taylor(history::History)
    agent_ids = vcat((history .|> collect)...) |> unique |> sort
    max_id = agent_ids |> maximum
    dict = OrderedDict{Int,Vector{Int}}()
    for aid in agent_ids
        dict[aid] = []
    end

    g = Graph(max_id)
    for (src, dst) in history
        add_edge!(g, src, dst)

        push!(dict[src], degree(g, src))
        push!(dict[dst], degree(g, dst))
    end

    function k(u)
        __v = Int[]
        for _v in dict |> values
            if length(_v) >= u
                push!(__v, _v[u])
            end
        end
        return __v
    end

    max_u = dict |> values .|> length |> maximum
    μ = [k(u) for u in 1:max_u] .|> mean
    σ = [k(u) for u in 1:max_u] .|> var

    data = DataFrame(filter(row -> all(v -> v > 0, row), eachrow(DataFrame(; μ, σ))))

    x = data.μ
    y = data.σ

    X = log10.(x)
    Y = log10.(y)

    _data = DataFrame(; X, Y)
    ols = lm(@formula(Y ~ X), _data)

    return coef(ols)[2], (x=data.μ, y=data.σ, predicted=predict(ols))
end

function calc_jsd(history1::History, history2::History)
    # 定数
    nbin = 50

    birthstep = begin
        # edges = exp10.(range(0; stop=log10(length(history1)), length=nbin))
        edges = range(0, length(history1), nbin)

        birthsteps1 = get_birthsteps(history1) |> values |> collect
        birthsteps2 = get_birthsteps(history2) |> values |> collect

        h1 = normalize(fit(Histogram, birthsteps1, edges); mode=:probability)
        h2 = normalize(fit(Histogram, birthsteps2, edges); mode=:probability)

        P = P_tmn = h1.weights
        Q = P_model = h2.weights
        M = (P + Q) / 2

        D_PM = kldivergence(P, M, 2)
        D_QM = kldivergence(Q, M, 2)

        (D_PM + D_QM) / 2
    end

    active_freq = begin
        flatten_history1 = vcat((history1 .|> collect)...)
        flatten_history2 = vcat((history2 .|> collect)...)

        counts1 = countmap(flatten_history1) |> values |> collect
        counts2 = countmap(flatten_history2) |> values |> collect

        active_freq1 = counts1 ./ maximum(counts1)
        active_freq2 = counts2 ./ maximum(counts2)

        # edges = exp10.(range(minimum(active_freq1) |> log10; stop=0, length=nbin))
        edges = range(minimum(active_freq1), 1, nbin)

        h1 = normalize(fit(Histogram, active_freq1, edges); mode=:probability)
        h2 = normalize(fit(Histogram, active_freq2, edges); mode=:probability)

        P = P_tmn = h1.weights
        Q = P_model = h2.weights
        M = (P + Q) / 2

        D_PM = kldivergence(P, M, 2)
        D_QM = kldivergence(Q, M, 2)

        (D_PM + D_QM) / 2
    end

    return (; birthstep, active_freq)
end