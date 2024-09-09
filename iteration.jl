using KitBase.Threads

include("basic_algo.jl")
include("variable_trans.jl")

VanLeer(sL, sR) =
    (fortsign(1.0, sL) + fortsign(1.0, sR)) * abs(sL) * abs(sR) /
    (abs(sL) + abs(sR) + 1.e-7)

fortsign(x, y) = abs(x) * sign(y)

sound_speed(prim, γ) = (0.5 * γ / prim[end])^0.5

function reconstruct!(ks, ctr)
    nx, ny, dx, dy = ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy
    # 对于边界进行处理
    @inbounds @threads for j = 1:ny
        swL = @view ctr[1, j].sw[:, 1]
        swL .= (ctr[2, j].w - ctr[1, j].w) / 0.5 * (dx[1, j] + dx[2, j])
        swR = @view ctr[nx, j].sw[:, 1]
        swR .= (ctr[nx, j].w - ctr[nx-1, j].w) / 0.5 * (dx[nx-1, j] + dx[nx, j])
    end

    @inbounds @threads for i = 1:nx
        swL = @view ctr[i, 1].sw[:, 2]
        swL .= (ctr[i, 2].w - ctr[i, 1].w) / 0.5 * (dx[i, 1] + dx[i, 2])
        swR = @view ctr[i, ny].sw[:, 2]
        swR .= (ctr[i, nx].w - ctr[i, nx-1].w) / 0.5 * (dx[i, nx-1] + dx[i, nx])
    end

    # 内部sw值的修改，使用是van leer
    @inbounds @threads for j = 1:ny
        for i = 2:nx-1
            sw = @view ctr[i, j].sw[:, 1]; 
            sL = ctr[i, j].w - ctr[i-1, j].w / 0.5 * (dx[i-1, j] + dx[i, j])            
            sR = ctr[i+1, j].w - ctr[i, j].w / 0.5 * (dx[i+1, j] + dx[i, j])
            sw .= VanLeer.(sL, sR)
        end
    end

    @inbounds @threads for j = 2:ny-1
        for i = 1:nx
            sw = @view ctr[i, j].sw[:, 2]; 
            sD = ctr[i, j].w - ctr[i, j-1].w / 0.5 * (dy[i, j-1] + dy[i, j])            
            sU = ctr[i, j+1].w - ctr[i, j].w / 0.5 * (dy[i, j+1] + dy[i, j])
            sw .= VanLeer.(sD, sU)
        end
    end

    println("reconstruct work")

    return nothing
end


function evolve!(ks, ctr, a1face, a2face, dt, bc)
    nx, ny, dx, dy = 
        ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy

    # x direction
    @inbounds @threads for j = 1:ny
        for i = 2:nx
            n = ks.ps.n[i-1, j, 2]
            len = ks.ps.areas[i-1, j, 3]
            ctrL = ctr[i-1, j]
            ctrR = ctr[i, j]
            shL = ctrL.sh[:, :, 1]
            sbL = ctrL.sh[:, :, 1]
            shR = ctrR.sh[:, :, 1]
            sbR = ctrR.sb[:, :, 1]
            dxL = 0.5 .* dx[i-1, j]
            dxR = 0.5 .* dx[i, j]
            flux_kfvs!(
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                ctrL.h .+ shL .* dxL,
                ctrL.b .+ sbL .* dxL,
                ctrR.h .- shR .* dxR,
                ctrR.b .- sbR .* dxR,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                dt,
                len,
                shL,
                sbL,
                shR,
                sbR,
            )
        end
    end

    # y direction
    @inbounds @threads for j = 2:ny
        for i = 1:nx
            n = ks.ps.n[i, j-1, 3]
            len = ks.ps.areas[i, j-1, 3]
            ctrD = ctr[i, j-1]
            ctrU = ctr[i, j]
            shD = ctrD.sh[:, :, 1]
            sbD = ctrD.sh[:, :, 1]
            shU = ctrU.sh[:, :, 1]
            sbU = ctrU.sb[:, :, 1]
            dxD = 0.5 .* dy[i, j-1]
            dxU = 0.5 .* dy[i, j]
            flux_kfvs!(
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                ctrD.h .+ shD .* dxD,
                ctrD.b .+ sbD .* dxD,
                ctrU.h .- shU .* dxU,
                ctrU.b .- sbU .* dxU,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                dt,
                len,
                shD,
                sbD,
                shU,
                sbU,
            )
        end
    end

    evolve_boundary!(ks, ctr, a1face, a2face, dt, bc)

    return nothing
end

function evolve_boundary!(
    ks,
    ctr,
    a1face,
    a2face,
    dt,
    boundary,
)
    println("evolve boundary is working")
    nx, ny, dx, dy = 
            ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy
    println("boundary is ", boundary)
    boundarys = ifelse(boundary isa Symbol, [boundary, boundary, boundary, boundary], boundary)
    println("boundarys[1] is ", boundarys[1])
    if boundarys[1] == "maxwell"
        println("if work")
        @inbounds @threads for j = 1:ny
            println("threads work")
            n = -ks.ps.n[1, j, 4]
            vn, vt = local_velocity(ks.vs.u, ks.vs.v, n...)
            xc = (ks.ps.vertices[1, j, 1, 1] + ks.ps.vertices[1, j, 4, 1]) / 2
            yc = (ks.ps.vertices[1, j, 1, 2] + ks.ps.vertices[1, j, 4, 2]) / 2
            bcL = local_frame(ks.init_bound.bc(xc, yc, ks.init_bound.p), n...)
            flux_boundary_maxwell!(
                a1face[1, j].fw,
                a1face[1, j].fh,
                a1face[1, j].fb,
                bcL, # left
                ctr[1, j].h,
                ctr[1, j].b,
                vn,
                vt,
                ks.vs.weights,
                ks.gas.K,
                dt,
                dy[1, j],
                1,
            )
            a1face[1, j].fw .= global_frame(a1face[1, j].fw, n...)
        end
    else
        throw(ArgumentError("boundary is not \"maxwell\", got $(boundarys[1])"))
    end

    if boundarys[2] == "maxwell"
        @inbounds @threads for j = 1:ny
            n = ks.ps.n[nx, j, 2]
            vn, vt = local_velocity(ks.vs.u, ks.vs.v, n...)
            xc = (ks.ps.vertices[nx, j, 2, 1] + ks.ps.vertices[nx, j, 3, 1]) / 2
            yc = (ks.ps.vertices[nx, j, 2, 2] + ks.ps.vertices[nx, j, 3, 2]) / 2
            bcR = local_frame(ks.init_bound.bc(xc, yc, ks.init_bound.p), n...)
            flux_boundary_maxwell!(
                a1face[nx+1, j].fw,
                a1face[nx+1, j].fh,
                a1face[nx+1, j].fb,
                bcR, # right
                ctr[nx, j].h,
                ctr[nx, j].b,
                vn,
                vt,
                ks.vs.weights,
                ks.gas.K,
                dt,
                dy[nx, j],
                -1,
            )
            a1face[nx+1, j].fw .= global_frame(a1face[nx+1, j].fw, n...)
        end
    else
        throw(ArgumentError("boundary is not \"maxwell\", got $(boundarys[2])"))
    end

    if boundarys[3] == "maxwell"
        @inbounds @threads for i = 1:nx
            n = -ks.ps.n[i, 1, 1]
            vn, vt = local_velocity(ks.vs.u, ks.vs.v, n...)
            xc = (ks.ps.vertices[i, 1, 1, 1] + ks.ps.vertices[i, 1, 2, 1]) / 2
            yc = (ks.ps.vertices[i, 1, 1, 2] + ks.ps.vertices[i, 1, 2, 2]) / 2
            bcD = local_frame(ks.init_bound.bc(xc, yc, ks.init_bound.p), n...)

            flux_boundary_maxwell!(
                a2face[i, 1].fw,
                a2face[i, 1].fh,
                a2face[i, 1].fb,
                bcD, # down
                ctr[i, 1].h,
                ctr[i, 1].b,
                vn,
                vt,
                ks.vs.weights,
                ks.gas.K,
                dt,
                dx[i, 1],
                1,
            )
            a2face[i, 1].fw .= global_frame(a2face[i, 1].fw, n...)
        end
    else
        throw(ArgumentError("boundary is not \"maxwell\", got $(boundarys[3])"))
    end

    if boundarys[4] == "maxwell"
        @inbounds @threads for i = 1:nx
            n = ks.ps.n[i, ny, 3]
            vn, vt = local_velocity(ks.vs.u, ks.vs.v, n...)
            xc = (ks.ps.vertices[i, ny, 3, 1] + ks.ps.vertices[i, ny, 4, 1]) / 2
            yc = (ks.ps.vertices[i, ny, 3, 2] + ks.ps.vertices[i, ny, 4, 2]) / 2
            bcU = local_frame(ks.init_bound.bc(xc, yc, ks.init_bound.p), n...)

            flux_boundary_maxwell!(
                a2face[i, ny+1].fw,
                a2face[i, ny+1].fh,
                a2face[i, ny+1].fb,
                bcU, # up
                ctr[i, ny].h,
                ctr[i, ny].b,
                vn,
                vt,
                ks.vs.weights,
                ks.gas.K,
                dt,
                dx[i, ny],
                -1,
            )
            a2face[i, ny+1].fw .= global_frame(a2face[i, ny+1].fw, n...)
        end
    else
        throw(ArgumentError("boundary is not \"maxwell\", got $(boundarys[1])"))
    end

    return nothing

end

"""
2D2F2V
"""
function flux_boundary_maxwell!(fw, fh, fb, bc, h, b, u, v, ω, inK, dt, len, rot)
    println("flux_boundary_maxwell! work")
    @assert length(bc) == 4
    println("bc is ", bc)
    δ = heaviside.(u .* rot)

    SF = sum(ω .* u .* h .* (1.0 .- δ))
    SG =
        (bc[end] / π) *
        sum(ω .* u .* exp.(-bc[end] .* ((u .- bc[2]) .^ 2 .+ (v .- bc[3]) .^ 2)) .* δ)
    prim = [-SF / SG; bc[2:end]]

    H = maxwellian(u, v, prim...)
    B = H .* inK ./ (2.0 * prim[end])

    hWall = H .* δ .+ h .* (1.0 .- δ)
    bWall = B .* δ .+ b .* (1.0 .- δ)

    fw[1] = discrete_moments(hWall, u, ω, 1) * len * dt
    fw[2] = discrete_moments(hWall, u, ω, 2) * len * dt
    fw[3] = discrete_moments(hWall .* u, v, ω, 1) * len * dt
    fw[4] =
        (
            0.5 * discrete_moments(hWall .* (u .^ 2 .+ v .^ 2), u, ω, 1) +
            0.5 * discrete_moments(bWall, u, ω, 1)
        ) *
        len *
        dt

    @. fh = u * hWall * len * dt
    @. fb = u * bWall * len * dt

    return nothing

end

function timestep(ks, ctr, simTime)
    nx, ny, dx, dy = 
        ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy

    tmax = 0.0

    @inbounds @threads for j = 1:ny
        for i = 1:nx
            prim = ctr[i, j].prim
            sos = sound_speed(prim, ks.gas.γ)
            umax, vmax = 
                max(ks.vs.u1, abs(prim[2])) + sos, max(ks.vs.v1, abs(prim[3])) + sos
            tmax = max(tmax, umax / dx[i, j] + vmax / dy[i, j])
        end
    end

    dt = ks.set.cfl / tmax
    dt = ifelse(dt < (ks.set.maxTime - simTime), dt, ks.set.maxTime - simTime)
    println("timestep work")

    return dt
end

function flux_kfvs!(fw, fh, fb, hL, bL, hR, bR, u, v, ω, dt, len, shL, sbL, shR, sbR)
    #--- upwind reconstruction ---#
    δ = heaviside.(u)

    h = @. hL * δ + hR * (1.0 - δ)
    b = @. bL * δ + bR * (1.0 - δ)
    sh = @. shL * δ + shR * (1.0 - δ)
    sb = @. sbL * δ + sbR * (1.0 - δ)
    @. fh = (dt * u * h - 0.5 * dt^2 * u^2 * sh) * len
    @. fb = (dt * u * b - 0.5 * dt^2 * u^2 * sb) * len

    fw[1] = sum(ω .* fh)
    fw[2] = sum(u .* ω .* fh)
    fw[3] = sum(v .* ω .* fh)
    fw[end] = 0.5 * (sum((u .^ 2 .+ v .^ 2) .* ω .* fh) + sum(ω .* fb))
end


function update!(
    ks,
    ctr,
    a1face,
    a2face,
    dt,
    residual;
    coll,
    bc,
)

    nx, ny, dx, dy = 
            ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy

    sumRes = zero(ctr[1].w)
    sumAvg = zero(ctr[1].w)

    @inbounds @threads for j = 2:ny-1
        for i = 2:nx-1
            step!(
                ctr[i,j].w,
                ctr[i,j].prim,
                ctr[i,j].h,
                ctr[i,j].b,
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                a1face[i+1, j].fw,
                a1face[i+1, j].fh,
                a1face[i+1, j].fb,
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                a2face[i, j+1].fw,
                a2face[i, j+1].fh,
                a2face[i, j+1].fb,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                dx[i, j] * dy[i, j],
                dt,
                sumRes,
                sumAvg,
                coll,
            )
        end
    end

    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * nx * ny) / (sumAvg[i] + 1.e-7)
    end

    update_boundary!(
        ks,
        ctr,
        a1face,
        a2face,
        dt,
        residual,
        ks.set.collision,
        bc,
    )

    return nothing

end



function update_boundary!(
    ks,
    ctr,
    a1face,
    a2face,
    dt,
    residual,
    coll,
    bc,
)
    bcs = ifelse(bc isa Symbol, [bc, bc, bc, bc], bc)

    nx, ny, dx, dy =
            ks.ps.nx, ks.ps.ny, ks.ps.dx, ks.ps.dy

    resL = zero(ctr[1].w)
    avgL = zero(ctr[1].w)
    resR = zero(ctr[1].w)
    avgR = zero(ctr[1].w)
    resU = zero(ctr[1].w)
    avgU = zero(ctr[1].w)
    resD = zero(ctr[1].w)
    avgD = zero(ctr[1].w)

    if bcs[1] != :fix
        @inbounds for j = 1:ny
            step!(
                ctr[1,j].w,
                ctr[1,j].prim,
                ctr[1,j].h,
                ctr[1,j].b,
                a1face[1, j].fw,
                a1face[1, j].fh,
                a1face[1, j].fb,
                a1face[2, j].fw,
                a1face[2, j].fh,
                a1face[2, j].fb,
                a2face[1, j].fw,
                a2face[1, j].fh,
                a2face[1, j].fb,
                a2face[1, j+1].fw,
                a2face[1, j+1].fh,
                a2face[1, j+1].fb,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                dx[1, j] * dy[1, j],
                dt,
                resL,
                avgL,
                coll,
            )
        end
    end

    if bcs[2] != :fix
        @inbounds for j = 1:ny
            step!(
                ctr[nx,j].w,
                ctr[nx,j].prim,
                ctr[nx,j].h,
                ctr[nx,j].b,
                a1face[nx, j].fw,
                a1face[nx, j].fh,
                a1face[nx, j].fb,
                a1face[nx+1, j].fw,
                a1face[nx+1, j].fh,
                a1face[nx+1, j].fb,
                a2face[nx, j].fw,
                a2face[nx, j].fh,
                a2face[nx, j].fb,
                a2face[nx, j+1].fw,
                a2face[nx, j+1].fh,
                a2face[nx, j+1].fb,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                dx[nx, j] * dy[nx, j],
                dt,
                resR,
                avgR,
                coll,
            )
        end
    end

    if bcs[3] != :fix
        @inbounds for i = 2:nx-1 # skip overlap
            step!(
                ctr[i,1].w,
                ctr[i,1].prim,
                ctr[i,1].h,
                ctr[i,1].b,
                a1face[i, 1].fw,
                a1face[i, 1].fh,
                a1face[i, 1].fb,
                a1face[i+1, 1].fw,
                a1face[i+1, 1].fh,
                a1face[i+1, 1].fb,
                a2face[i, 1].fw,
                a2face[i, 1].fh,
                a2face[i, 1].fb,
                a2face[i, 2].fw,
                a2face[i, 2].fh,
                a2face[i, 2].fb,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                dx[i, 1] * dy[i, 1],
                dt,
                resD,
                avgD,
                coll,
            )
        end
    end

    if bcs[4] != :fix
        @inbounds for i = 2:nx-1 # skip overlap
            step!(
                ctr[i,ny].w,
                ctr[i,ny].prim,
                ctr[i,ny].h,
                ctr[i,ny].b,
                a1face[i, ny].fw,
                a1face[i, ny].fh,
                a1face[i, ny].fb,
                a1face[i+1, ny].fw,
                a1face[i+1, ny].fh,
                a1face[i+1, ny].fb,
                a2face[i, ny].fw,
                a2face[i, ny].fh,
                a2face[i, ny].fb,
                a2face[i, ny+1].fw,
                a2face[i, ny+1].fh,
                a2face[i, ny+1].fb,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                dx[i, ny] * dy[i, ny],
                dt,
                resD,
                avgD,
                coll,
            )
        end
    end

    for i in eachindex(residual)
        residual[i] +=
            sqrt((resL[i] + resR[i] + resU[i] + resD[i]) * 2) /
            (avgL[i] + avgR[i] + avgU[i] + avgD[i] + 1.e-7)
    end

    """ngx = 1 - first(eachindex(ks.ps.x[:, 1]))
    if bcs[1] == :period
        bc_period!(ctr, ngx; dirc = :x)
    elseif bcs[1] in (:extra, :mirror)
        bcfun = eval(Symbol("bc_" * string(bcs[1]) * "!"))
        bcfun(ctr, ngx; dirc = :xl)
    end
    if bcs[2] in (:extra, :mirror)
        bcfun = eval(Symbol("bc_" * string(bcs[2]) * "!"))
        bcfun(ctr, ngx; dirc = :xr)
    end

    ngy = 1 - first(eachindex(ks.ps.y[1, :]))
    if bcs[3] == :period
        bc_period!(ctr, ngy; dirc = :y)
    elseif bcs[3] in (:extra, :mirror)
        bcfun = eval(Symbol("bc_" * string(bcs[3]) * "!"))
        bcfun(ctr, ngy; dirc = :yl)
    end
    if bcs[4] in (:extra, :mirror)
        bcfun = eval(Symbol("bc_" * string(bcs[4]) * "!"))
        bcfun(ctr, ngy; dirc = :yr)
    end"""

    return nothing

end

"""
2D2F2V
"""
function step!(w, prim, h, b, fwL, fhL, fbL, fwR, fhR, fbR, fwD,
    fhD, fbD, fwU, fhU, fbU, u, v, weights, K, γ, μᵣ, ω, Pr, Δs,
    dt, RES, AVG, collision)
    #--- store W^n and calculate shakhov term ---#
    w_old = deepcopy(w)

    if collision == :shakhov
        q = heat_flux(h, b, prim, u, v, weights)

        MH_old = maxwellian(u, v, prim...)
        MB_old = MH_old .* K ./ (2.0 * prim[end])
        SH, SB = shakhov(u, v, MH_old, MB_old, q, prim, Pr, K)
    else
        SH = zero(h)
        SB = zero(b)
    end

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR + fwD - fwU) / Δs
    prim = similar(w)
    conserve_prim!(prim, w, γ)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- calculate M^{n+1} and tau^{n+1} ---#
    MH = maxwellian(u, v, prim...)
    MB = MH .* K ./ (2.0 * prim[end])
    MH .+= SH
    MB .+= SB
    
    τ = vhs_collision_time(prim[1], prim[4], μᵣ, ω)

    #--- update distribution function ---#
    for i in eachindex(u)
        h[i] =
            (h[i] + (fhL[i] - fhR[i] + fhD[i] - fhU[i]) / Δs + dt / τ * MH[i]) /
            (1.0 + dt / τ)
        b[i] =
            (b[i] + (fbL[i] - fbR[i] + fbD[i] - fbU[i]) / Δs + dt / τ * MB[i]) /
            (1.0 + dt / τ)
    end

end