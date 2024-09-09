using KitBase.OffsetArrays,KitBase.LinearAlgebra
using Base: @kwdef
include("basic_algo.jl")
include("variable_trans.jl")

struct PSpace2D{TR<:Real,TI<:Integer,TA,TB,TC,TD}
 x0::TR
 x1::TR
 nx::TI
 y0::TR
 y1::TR
 ny::TI
 x::TA
 y::TA
 dx::TA
 dy::TA
 vertices::TB
 areas::TC
 n::TD
end

function PSpace2D(
    X0::TR,
    X1::TR,
    NX::TI,
    Y0::TR,
    Y1::TR,
    NY::TI,
    NGX = 0::Integer,
    NGY = 0::Integer,
) where {TR,TI}
    TX = ifelse(TR == Float32, Float32, Float64)

    δx = (X1 - X0) / NX
    δy = (Y1 - Y0) / NY
    x = OffsetArray{TX}(undef, 1-NGX:NX+NGX, 1-NGY:NY+NGY)
    y = similar(x)
    dx = similar(x)
    dy = similar(x)
    for j in axes(x, 2)
        for i in axes(x, 1)
            x[i, j] = X0 + (i - 0.5) * δx
            y[i, j] = Y0 + (j - 0.5) * δy
            dx[i, j] = δx
            dy[i, j] = δy
        end
    end

    vertices = similar(x, axes(x)..., 4, 2)
    for j in axes(vertices, 2), i in axes(vertices, 1)
        vertices[i, j, 1, 1] = x[i, j] - 0.5 * dx[i, j]
        vertices[i, j, 2, 1] = x[i, j] + 0.5 * dx[i, j]
        vertices[i, j, 3, 1] = x[i, j] + 0.5 * dx[i, j]
        vertices[i, j, 4, 1] = x[i, j] - 0.5 * dx[i, j]

        vertices[i, j, 1, 2] = y[i, j] - 0.5 * dy[i, j]
        vertices[i, j, 2, 2] = y[i, j] - 0.5 * dy[i, j]
        vertices[i, j, 3, 2] = y[i, j] + 0.5 * dy[i, j]
        vertices[i, j, 4, 2] = y[i, j] + 0.5 * dy[i, j]
    end

    areas = [0.0 for i in axes(x, 1), j in axes(x, 2), k = 1:4]
    for j in axes(x, 2), i in axes(x, 1)
        areas[i, j, 1] = point_distance(vertices[i, j, 1, :], vertices[i, j, 2, :])
        areas[i, j, 2] = point_distance(vertices[i, j, 2, :], vertices[i, j, 3, :])
        areas[i, j, 3] = point_distance(vertices[i, j, 3, :], vertices[i, j, 4, :])
        areas[i, j, 4] = point_distance(vertices[i, j, 4, :], vertices[i, j, 1, :])
    end

    n = [zeros(2) for i in axes(x, 1), j in axes(x, 2), k = 1:4]
    for j in axes(x, 2), i in axes(x, 1)
        n1 = unit_normal(vertices[i, j, 1, :], vertices[i, j, 2, :])
        n1 .= ifelse(
            dot(
                n1,
                [x[i, j], y[i, j]] .- (vertices[i, j, 1, :] .+ vertices[i, j, 2, :]) ./ 2,
            ) <= 0,
            n1,
            -n1,
        )

        n2 = unit_normal(vertices[i, j, 2, :], vertices[i, j, 3, :])
        n2 .= ifelse(
            dot(
                n2,
                [x[i, j], y[i, j]] .- (vertices[i, j, 2, :] .+ vertices[i, j, 3, :]) ./ 2,
            ) <= 0,
            n2,
            -n2,
        )

        n3 = unit_normal(vertices[i, j, 3, :], vertices[i, j, 4, :])
        n3 .= ifelse(
            dot(
                n3,
                [x[i, j], y[i, j]] .- (vertices[i, j, 3, :] .+ vertices[i, j, 4, :]) ./ 2,
            ) <= 0,
            n3,
            -n3,
        )

        n4 = unit_normal(vertices[i, j, 4, :], vertices[i, j, 1, :])
        n4 .= ifelse(
            dot(
                n4,
                [x[i, j], y[i, j]] .- (vertices[i, j, 4, :] .+ vertices[i, j, 1, :]) ./ 2,
            ) <= 0,
            n4,
            -n4,
        )

        n[i, j, 1] .= n1
        n[i, j, 2] .= n2
        n[i, j, 3] .= n3
        n[i, j, 4] .= n4
    end

    return PSpace2D{TR,TI,typeof(x),typeof(vertices),typeof(areas),typeof(n)}(
        X0,
        X1,
        NX,
        Y0,
        Y1,
        NY,
        x,
        y,
        dx,
        dy,
        vertices,
        areas,
        n,
    )
end

struct VSpace2D{TR,TI,TA}
    u0::TR
    u1::TR
    nu::TI
    v0::TR
    v1::TR
    nv::TI
    u::TA
    v::TA
    du::TA
    dv::TA
    weights::TA
end

function VSpace2D(
    U0,
    U1,
    NU::TI,
    V0,
    V1,
    NV::TI;
    precision = Float64,
) where {TI<:Integer}

    δu = (U1 - U0) / NU
    δv = (V1 - V0) / NV
    u = Array{precision}(undef, NU, NV)
    v = similar(u)
    du = similar(u)
    dv = similar(u)
    weights = similar(u)

    for j in axes(u, 2)
        for i in axes(u, 1)
            u[i, j] = U0 + (i - 0.5) * δu
            v[i, j] = V0 + (j - 0.5) * δv
            du[i, j] = δu
            dv[i, j] = δv
            weights[i, j] = δu * δv
        end
    end

    return VSpace2D{precision,TI,typeof(u)}(U0, U1, NU, V0, V1, NV, u, v, du, dv, weights)

end

@kwdef mutable struct Gas{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11<:Integer,T12}
    Kn::T1 = 1e-2
    Ma::T2 = 0.0
    Pr::T3 = 1.0
    K::T4 = 2.0
    γ::T5 = 5 / 3
    ω::T6 = 0.81
    αᵣ::T7 = 1.0
    ωᵣ::T8 = 0.5
    μᵣ::T9 = ref_vhs_vis(Kn, αᵣ, ωᵣ)
    m::T10 = 1e-3
    np::T11 = 1000
    fsm::T12 = nothing
end

function ref_vhs_vis(Kn, alpha, omega)
    5.0 * (alpha + 1.0) * (alpha + 2.0) * √π /
    (4.0 * alpha * (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega)) * Kn
end

struct SolverSet{TS,TP,TV,TG,Ti}
    # setup
    set::TS
    # physical space
    ps::TP
    # velocity space
    vs::TV
    # gas property
    gas::TG
    # initial and boundary
    init_bound::Ti
end

"""function SolverSet(set, ps, vs, gas, init_bound)
    return SolverSet{}(set, ps, vs, gas, init_bound)
end"""

function init_bound_cavity(set, ps, vs, gas, Um = 0.15, Vm = 0.0, Tm = 1.0)
    prim = [1.0, 0.0, 0.0, 1.0]
    w = similar(prim)
    prim_conserve!(prim, w, gas.γ)
    primU = [1.0, Um, Vm, Tm]

    p = (y1 = ps.y1, w = w, prim = prim, primU = primU)

    fw = function (args...)
        p = args[end]
        return p.w
    end

    bc = function (x, y, args...)
        p = args[end]

        if y == p.y1
            return p.primU
        else
            return p.prim
        end
    end

    h = maxwellian(vs.u, vs.v, prim...)
    b = h .* gas.K / 2.0 / prim[end]
    p = (p..., h = h, b = b)
    ff = function (args...)
        p = args[end]
        return p.h, p.b
    end

    return fw, ff, bc, p
end

mutable struct IB2F{TF1,TF2,T,NT}
    fw::TF1
    ff::TF2
    bc::T
    p::NT
end

"""
Initialize finite volume data structure
"""
function init_fvm(ks)
    nx, ny, dx, dy = ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy

    ctr = OffsetArray{ControlVolume2F}(undef, axes(ks.ps.x, 1), axes(ks.ps.y, 2))
    a1face = Array{Interface2F}(undef, nx + 1 , ny)
    a2face = Array{Interface2F}(undef, nx, ny + 1)
    # 网格赋初值
    for j in axes(ctr, 2), i in axes(ctr, 1)
        prim = [1.0, 0.0, 0.0, 1.0]
        w = similar(prim)
        prim_conserve!(prim, w, ks.gas.γ)
        sw = [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
        h = maxwellian(ks.vs.u, ks.vs.v, prim...)
        b = h .* ks.gas.K / 2.0 / prim[end]
        sh = similar(h, size(h)..., 2)
        sh .= 0
        sb = similar(b, size(b)..., 2)
        sb .= 0
        ctr[i, j] = ControlVolume2F(w, prim, sw, h, b, sh, sb)
    end
    
    fw = [1.0, 0.0, 0.0, 0.749999999]
    fh = similar(ctr[1,1].h)
    fh .= 0
    fb = similar(ctr[1,1].b)
    fb .= 0
    # 界面赋初值，先是沿x轴的方向
    for j = 1:ny
        for i = 1:nx+1
            a1face[i, j] = Interface2F(fw, fh, fb)
        end
    end
    for i = 1:nx
        for j = 1:ny+1
            a2face[i, j] = Interface2F(fw, fh, fb)
        end
    end
    return ctr, a1face, a2face
end

struct Interface2F
    fw::Vector{Float64}
    fh::Matrix{Float64}
    fb::Matrix{Float64}
end

"""function Interface2F(fw, fh, fb)
    n = size(fw, 1) - 2
    return Interface2F{}(fw, fh, fb)
end
"""

struct ControlVolume2F
    w::Vector{Float64}
    prim::Vector{Float64}
    sw::Matrix{Float64}
    h::Matrix{Float64}
    b::Matrix{Float64}
    sh::Array{Float64,3}
    sb::Array{Float64,3}
end

"""function ControlVolume2F(w, prim, sw, h, b, sh, sb)
    return ControlVolume2F{}(
        w,
        prim,
        sw,
        h,
        b,
        sh,
        sb,
    )
end"""

"""function ControlVolume2F(w, prim, h, b)
    sw = zero(w)
    sh = zero(h)
    sb = zero(b)
    return ControlVolume2F{}(
        w,
        prim,
        sw,
        h,
        b,
        sh,
        sb,
    )
end"""

@kwdef struct Setup{S,I<:Integer,E,F<:Real,G<:Real}
    matter::S = "gas"
    case::S = "dev"
    space::S = "1d0f0v"
    flux::S = "kfvs"
    collision::S = "bgk"
    nSpecies::I = 1
    interpOrder::I = 2
    limiter::S = "vanleer"
    boundary::E = ["fix", "fix"]
    cfl::F = 0.5
    maxTime::G = 0.1
    hasForce::Bool = false
end

function Setup(
    matter,
    case,
    space,
    flux,
    collision,
    ns,
    order,
    limiter,
    bc::T,
    cfl,
    time,
    hasForce = false,
) where {T}
    boundary = begin
        if parse(Int, space[1]) == 1
            [bc, bc]
        elseif parse(Int, space[1]) == 2
            [bc, bc, bc, bc]
        end
    end

    return Setup{typeof(matter),typeof(ns),typeof(boundary),typeof(cfl),typeof(time)}(
        matter,
        case,
        space,
        flux,
        collision,
        ns,
        order,
        limiter,
        boundary,
        cfl,
        time,
        hasForce,
    )
end