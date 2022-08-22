include("Types.jl")
include("Utils.jl")

using PlotlyJS, DataFrames, GLM

"""
エージェントの誕生ステップとアクセス数の関係をプロットする
"""
function plot_time_access_scatter(history::History)
    flatten_history = vcat((history .|> collect)...)

    birthsteps = get_birthsteps(history) |> sort |> values |> collect
    counts = countmap(flatten_history) |> sort |> values |> collect

    pltdata = scatter(; x=birthsteps, y=counts ./ length(history), mode="markers")
    layout = Layout(;
        template=templates[:simple_white],
        xaxis=attr(; type=:log, title="Birth step"),
        yaxis=attr(; type=:log, title="Active frequency"),
    )
    return plot(pltdata, layout)
end

function plot_rich_get_richer_triangle(history::History, tau::Int=1000)
    separators = 1:tau:length(history)
    intervals = [history[separator:min(separator + tau, end)] for separator in separators]

    agent_birthsteps = get_birthsteps(history)
    birthsteps =
        [
            most_accessed_agent_birthstep_in_interval(interval, agent_birthsteps) for
            interval in intervals
        ] .|> last

    pltdata = scatter(;
        x=1:length(intervals),
        y=birthsteps,
        mode="markers",
        marker=attr(; opacity=0.75, size=16),
    )
    layout = Layout(;
        xaxis_title="Interval",
        yaxis_title="Birth step",
        yaxis_range=[1, length(history)],
        showlegend=false,
        template=templates[:simple_white],
    )
    return plot(pltdata, layout)
end

function plot_rich_get_richer_triangle_in_ratio(history::History, ratio=0.10; tau=1000)
    separators = 1:tau:length(history)
    intervals = [history[separator:min(separator + tau, end)] for separator in separators]

    agent_birthsteps = get_birthsteps(history)
    birthsteps = [
        sort_accessed_agent_birthstep_in_interval(interval, agent_birthsteps) for
        interval in intervals
    ]

    x = Int[]
    y = Int[]

    for (index, bts) in enumerate(birthsteps)
        l = convert(Int, floor(length(bts) * ratio))
        append!(x, fill(index, l))
        append!(y, bts[1:l])
    end

    pltdata = scatter(; x, y, mode="markers", marker=attr(; opacity=0.75, size=8))
    layout = Layout(;
        xaxis_title="interval",
        yaxis_title="birthstep",
        yaxis_range=[1, length(history)],
        showlegend=false,
        template=templates[:simple_white],
    )
    return plot(pltdata, layout)
end

function plot_heaps(history::History)
    steps = Int[]
    uagents = Int[]
    set = Set{Int}()

    for (index, (src, dst)) in enumerate(history)
        push!(set, src, dst)
        push!(steps, index)
        push!(uagents, length(set))
    end

    X = log10.(steps)
    Y = log10.(uagents)
    data = DataFrame(; X, Y)
    ols = lm(@formula(Y ~ X), data)

    pow10 = x -> 10 .^ x

    pltdata = [
        scatter(;
            x=pow10.(data.X), y=pow10.(data.Y), mode=:lines, name="model", line_color="red"
        )
        scatter(;
            x=pow10.(data.X),
            y=pow10.(predict(ols)),
            mode=:lines,
            name="predicted",
            line_color="blue",
        )
    ]
    layout = Layout(;
        template=templates[:ggplot2],
        xaxis_type=:log,
        yaxis_type=:log,
        xaxis_title="step (log scale)",
        yaxis_title="number of unique agents (log scale)",
        annotations=[
            attr(;
                xref="x domain",
                yref="y domain",
                x=0.99,
                y=0.01,
                text="γ = $(coef(ols)[2])",
                showarrow=false,
                bordercolor="black",
                bgcolor="white",
                borderpad=8,
            ),
        ],
        legend=attr(; x=0.01, y=0.99, bordercolor="black", borderwidth=1),
    )
    return plot(pltdata, layout)
end

function plot_zipf(history::History)
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

    predicted = 10 .^ predict(ols)

    pltdata = [
        scatter(; x, y, name="model", line_color="red")
        scatter(; x, y=predicted, name="prediced", line_color="blue")
    ]
    layout = Layout(;
        template=templates[:ggplot2],
        xaxis_type=:log,
        yaxis_type=:log,
        xaxis_title="rank of frequency (log scale)",
        yaxis_title="frequency (log scale)",
        annotations=[
            attr(;
                xref="x domain",
                yref="y domain",
                x=0.01,
                y=0.01,
                text="α = $(coef(ols)[2])",
                showarrow=false,
                bordercolor="black",
                bgcolor="white",
                borderpad=8,
            ),
        ],
        legend=attr(; x=0.99, y=0.99, bordercolor="black", borderwidth=1, xanchor=:right),
    )
    plot(pltdata, layout)
end
