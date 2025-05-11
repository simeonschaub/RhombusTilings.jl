using Distributions, LogExpFunctions

logpochhammer(x, n) = sum(k -> log(x + k), 0:(n - 1); init = zero(x))

function sample_D!(tmp, a, b, n)
    a′, b′ = Float64(a), Float64(b)
    p = view(tmp, 1:(n + 1))
    map!(p, 0:n) do k
        logpochhammer(a′, k) - logpochhammer(b′, k)
    end
    s = logsumexp(p)
    p .= exp.(p .- s)
    return rand(DiscreteNonParametric(0:n, p; check_args = false))
end

function hahn_markov_step!(Y, X, tmp; N, T, S)
    Y[:, 1] .= 0:(N - 1)
    for t in 1:T
        i = 0
        while (i += 1) ≤ N
            xᵢ, yᵢ = X[i, t + 1], Y[i, t]
            if xᵢ == yᵢ
                k = xᵢ
                l = 1
                i′ = i
                while (i′ += 1) ≤ N
                    xᵢ, yᵢ = X[i′, t + 1], Y[i′, t]
                    xᵢ == yᵢ == k + l || break
                    l += 1
                end
                ξ = sample_D!(tmp, k + T − t − S, k + 1, l)
                Y[i:(i + ξ - 1), t + 1] .= k:(k + ξ - 1)
                Y[(i + ξ):(i + l - 1), t + 1] .= (k + ξ + 1):(k + l)

                i = i′ - 1
            elseif xᵢ > yᵢ
                Y[i, t + 1] = xᵢ
            else
                Y[i, t + 1] = yᵢ
            end
        end
    end
    return Y
end

struct HahnPaths
    paths::Matrix{Int}
    N::Int
    T::Int
    S::Int
end

function sample_hahn_paths(N, T, S)
    X, Y = Matrix{Int}(undef, N, T + 1), Matrix{Int}(undef, N, T + 1)
    tmp = Vector{Float64}(undef, N + 1)
    X .= 0:(N - 1)
    for S in 0:(S - 1)
        hahn_markov_step!(Y, X, tmp; N, T, S)
        X, Y = Y, X
    end
    return HahnPaths(X, N, T, S)
end

function RhombusTiling((; paths, N, T, S)::HahnPaths)
    top_tiles = zeros(Int, T - S, S)
    builder = RhombusTilingBuilder{3, Int}()

    for i in 1:N
        x, y = 0, 0
        for t in 1:T
            if paths[i, t + 1] == paths[i, t]
                add_tile!(builder, (i - 1, x, y), (0x01, 0x03))
                y += 1
            else
                add_tile!(builder, (i - 1, x, y), (0x01, 0x02))
                view(top_tiles, 1:y, x + 1) .+= 0x01
                x += 1
            end
        end
    end
    for (I, z) in pairs(IndexCartesian(), top_tiles)
        add_tile!(builder, (z, I[2] - 1, I[1] - 1), (0x02, 0x03))
    end

    return RhombusTiling(builder, (N, S, T - S))
end
