point_distance(p1::T, p2::T) where T = norm(p1 .- p2)

function unit_normal(p1::T, p2::T) where T
    Δ = p2 .- p1
    l = norm(Δ) + 1e-6

    return [-Δ[2], Δ[1]] ./ l
end

maxwellian(u, v, ρ, U, V, λ) = @. ρ * (λ / π) * exp(-λ * ((u - U)^2 + (v - V)^2))

"""
Discrete moments of particle distribution
"""
discrete_moments(f, ω) = sum(@. ω * f)

discrete_moments(f, u, ω, n) = sum(@. ω * u^n * f)


function local_velocity(u, v, cosa, sina)
    vn = @. u * cosa + v * sina
end

function local_frame(w, cosa, sina)
    L = similar(w, Float64)

    L[1] = w[1]
    L[2] = w[2] * cosa + w[3] * sina
    L[3] = w[3] * cosa - w[2] * sina
    L[4] = w[4]

    return L
end

function global_frame(w, cosa, sina)
    G = similar(w, Float64)
    
    G[1] = w[1]
    G[2] = w[2] * cosa - w[3] * sina
    G[3] = w[2] * sina + w[3] * cosa
    G[4] = w[4]

    return G
end

heaviside(x) = ifelse(x >= 0, one(x), zero(x))

vhs_collision_time(ρ, λ, μᵣ, ω) = μᵣ * 2.0 * λ^(1.0 - ω) / ρ