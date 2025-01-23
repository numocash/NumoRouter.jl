init_opt_cache(R) = (R⁺=similar(R), ∇ϕR = similar(R))

function test_optimality_conditions_met(c, Δ, Λ, strategy; cache=nothing)
    R, γ = strategy.R, strategy.γ
    if isnothing(cache)
        cache = init_opt_cache(R)
    end
    R⁺ = cache.R⁺
    ∇ϕR = cache.∇ϕR
    n = length(c)

    # Primal feasibility
    R⁺ = R + γ * Δ - Λ
    pfeas = all(Δ .≥ 0) && all(Λ .≥ 0)
    ϕR = CR.ϕ(strategy)
    ϕR⁺ = CR.ϕ(strategy; R=R⁺)
    CR.∇ϕ!(∇ϕR, strategy; R=R⁺)
    strategy_sat = ≈(ϕR, ϕR⁺) && ϕR⁺ ≥ ϕR - sqrt(eps())

    opt = maximum(i -> γ * ∇ϕR[i] / c[i], 1:n) ≤ minimum(i -> ∇ϕR[i] / c[i], 1:n) + sqrt(eps())
    @test pfeas && strategy_sat && opt
end

# For univ3, we specialize to use the price impact function
function test_optimality_conditions_met(c, Δ, Λ, strategy::LogNormal)
    p_opt = c[1] / c[2]
    γ, q = strategy.γ, strategy.current_price
    
    # Mkt price in current interval
    if γ * q ≤ p_opt ≤ q / γ
        @test Δ[1] == Δ[2] == 0
        return
    end

    # Out of liquidity
    if p_opt > strategy.lower_ticks[1]
        λ = CR.forward_trade(Δ, strategy)
        @test λ ≈ Λ[1] && Λ[2] == 0
        return
    elseif p_opt < strategy.lower_ticks[end] && iszero(strategy.liquidity[end])
        λ = CR.forward_trade(Δ, strategy)
        @test λ ≈ Λ[2] && Λ[1] == 0
        return
    end
    
    # Mkt price in larger interval
    if q > p_opt
        price_impact₊(δ) = ForwardDiff.gradient(x->CR.forward_trade(x, strategy), δ)[1]
        @test isapprox(price_impact₊(Δ), p_opt, atol=1e-6)
        return
    else
        price_impact₋(δ) = ForwardDiff.gradient(x->CR.forward_trade(x, strategy), δ)[2]
        @test isapprox(price_impact₋(Δ), 1/p_opt, atol=1e-6)
        return
    end
end

@testset "Strategies" begin
@testset "arbitrage checks: two coins" begin
    Δ = MVector{2, Float64}(undef)
    Λ = MVector{2, Float64}(undef)
    
    n = 3
    Random.seed!(1234)
    γs = [rand() for i in 1:n]
    Rs = [rand(2)*10 for i in 1:n]
    νs = [@MVector rand(2) for i in 1:n]
    cache = init_opt_cache(Rs[1])

    @testset "product" begin
        equal_pool = ProductTwoCoin([1, 1], 1, [1, 2])

        # No arb in the fee-less case
        v = MVector(1.0, 1.0)
        find_arb!(Δ, Λ, equal_pool, v)
        @test iszero(Δ) && iszero(Λ)

        v .*= 2
        find_arb!(Δ, Λ, equal_pool, v)
        @test iszero(Δ) && iszero(Λ)

        # Easy arb in the fee-less case
        v[2] = 1
        find_arb!(Δ, Λ, equal_pool, v)
        @test Δ[1] ≈ 0 && Δ[2] ≈ sqrt(2) - 1
        @test Λ[1] ≈ 1 - sqrt(1/2) && Λ[2] ≈ 0

        # No-arb in the fee case
        @test length(ProductTwoCoin([1, 1], .9, [1, 2])) == 2
        @test_throws ArgumentError ProductTwoCoin([1, 1], .9, [1])

        for R in Rs, γ in γs, ν in νs
            strategy = ProductTwoCoin(R, γ, [1, 2])
            find_arb!(Δ, Λ, strategy, ν)
            test_optimality_conditions_met(ν, Δ, Λ, strategy; cache=cache)
        end

    end

    @testset "geo mean" begin
        ws = [(w1=rand(); [w1; 1-w1]) for _ in 1:n]
        for R in Rs, γ in γs, ν in νs, w in ws
            strategy = GeometricMeanTwoCoin(R, w, γ, [1, 2])
            find_arb!(Δ, Λ, strategy, ν)
            test_optimality_conditions_met(ν, Δ, Λ, strategy; cache=cache)
        end
    end


end

@testset "univ3" begin
    Δ = zeros(2)
    Λ = zeros(2)

    # UniV3 pool params
    current_price = 15.0
    lower_ticks = [30., 20, 10, 5]
    liquidity = [1.0, 2.0, 1.5, 0.0]
    Ai = [1, 2]
    
    @testset "no fees" begin
        γ = 1.0
        strategy = UniV3(current_price, lower_ticks, liquidity, γ, Ai)

        # Opt trade == 0
        v = [15.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # same interval (below), but opt trade ≂̸ 0
        v = [16.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # same interval (above), but opt trade ≂̸ 0
        v = [14.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # prev interval 
        v = [25.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # next interval 
        v = [7.5, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # Drain all liquidity, going up
        v = [4., 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # Drain all liquidity, going down
        v = [35., 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)
    end

    @testset "fees" begin
        γ = 0.997
        strategy = UniV3(current_price, lower_ticks, liquidity, γ, Ai)

        # Opt trade == 0 (no arb interval)
        v = [15.0*(1+γ)/2, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # same pool (below), but opt trade ≂̸ 0
        v = [16.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # same pool (above), but opt trade ≂̸ 0
        v = [14.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # prev pool 
        v = [25.0, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # next pool 
        v = [7.5, 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # drain all liquidity, going up
        v = [4., 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)

        # drain all liquidity, going down
        v = [35., 1.0]
        find_arb!(Δ, Λ, strategy, v)
        test_optimality_conditions_met(v, Δ, Λ, strategy)
        
    end

end
end