# include("initialization.jl")
include("struct.jl")
include("iteration.jl")
using KitBase.ProgressMeter: @showprogress
using Plots
# using KitBase

function main()
    set = Setup( 
        case = "cacity",
        space = "2d2f2v",
        boundary = ["maxwell", "maxwell", "maxwell", "maxwell"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 1,
    )
    ps = PSpace2D(0, 1, 5, 0, 1, 5)
    # println(vs.v .* n[2])
    vs = VSpace2D(-5, 5, 5, -5, 5, 5)
    # println(size(vs.u))
    val = unit_normal((0, 0), (1, 0))
"""    println(val)
    println(ps.vertices[1, 1, 1, :])
    println(ps.vertices[1, 1, 2, :])
    println(ps.vertices[1, 1, 3, :])
    println(ps.vertices[1, 1, 4, :])"""
    gas = Gas(Kn = 0.075, K = 1.0)
    ib = IB2F(init_bound_cavity(set, ps, vs, gas)...) 

    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, a1face, a2face = init_fvm(ks)
    t = 0.0
    dt = timestep(ks, ctr, 0.0)
    nt = ks.set.maxTime ÷ dt |> Int
    res = zeros(4)
    # println(dt)
    @showprogress for iter = 1:nt
        reconstruct!(ks, ctr)
        evolve!(ks, ctr, a1face, a2face, dt, ks.set.boundary)
        update!(ks, ctr, a1face, a2face, dt, res; coll = ks.set.collision, bc = ks.set.boundary)

        if maximum(res) < 1e-6
            break
        end
    end
    plot(ks, ctr)
    print("plot is over")
    # ctr, a1face, a2face = init_fvm(ks)
    # a1face指沿着x轴的面，a2face指沿着y轴的。

    # t = 0.0
    # dt = timestep(ks, ctr, 0.0)
    # nt = ks.set.maxTime ÷ dt |> Int
    # res = zeros(4)
    # cd(@__DIR__)
    # ks, ctr, a1face, a2face, simTime = initialize("cavity.txt")
    # simTime = solve!(ks, ctr, a1face, a2face, simTime)
    # plot(ks, ctr)
end

main()