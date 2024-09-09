"""
prim: 
conserve:
"""

function prim_conserve!(prim, conserve, γ)
    conserve[1] = prim[1]
    conserve[2] = prim[1] * prim[2]
    conserve[3] = prim[1] * prim[3]
    conserve[4] = 0.5 * prim[1] / prim[4] / (γ - 1.0) + 0.5 * prim[1] * (prim[2]^2 + prim[3]^2)
end

function conserve_prim!(conserve, prim, γ)
    prim[1] = conserve[1]
    prim[2] = conserve[2] / conserve[1]
    prim[3] = conserve[3] / conserve[1]
    prim[4] = 0.5 * conserve[1] / (γ - 1.0) / (conserve[4] - 0.5 * (conserve[2]^2 + conserve[3]^2) / conserve[1])
end