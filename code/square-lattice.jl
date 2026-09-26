using CUDA
using DelimitedFiles # read files 
using LinearAlgebra
using Random

function init_CDW_kernel!(nSt, dim, fil, Δ, nn)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        nn[idx] = fil + Δ * (mod(j + k, 2) - 0.5) * 2.0
    end
    return nothing
end

function init_hmn_self_consistent_kernel!(nSt, tnn, hmn, vnn, vList, nn, nnL, nnR, nnT, nnB)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        hmn[idx, nnR[idx]] = -tnn
        hmn[idx, nnT[idx]] = -tnn
        hmn[nnR[idx], idx] = -tnn
        hmn[nnT[idx], idx] = -tnn
        hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nnT[idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end

function init_hmn_kernel!(nSt, tnn, hmn, testField, nnR, nnT)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        hmn[idx, nnR[idx]] = -tnn
        hmn[idx, nnT[idx]] = -tnn
        hmn[nnR[idx], idx] = -tnn
        hmn[nnT[idx], idx] = -tnn
        hmn[idx, idx] = testField[idx]
    end
    return nothing
end

function updt_hmn_self_consistent_kernel!(nSt, hmn, vnn, vList, nn, nnL, nnR, nnT, nnB)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nnT[idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end

function updt_hmn_kernel!(nSt, hmn, testField)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        hmn[idx, idx] = testField[idx]
    end
    return nothing
end

function comp_fermi_kernel!(nSt, μ, kT, energy, fermiF)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        ratio = (energy[idx] - μ)/kT
        if ratio > 30.0
            ## fermiFactor ≈ 1E-14
            fermiF[idx] = 0.0
        elseif ratio < -30.0
            ## 1 - fermiFactor ≈ 1E-14
            fermiF[idx] = 1.0
        else
            fermiF[idx] = 1.0 / (exp(ratio) + 1.0)
        end
    end
    return nothing
end

# function comp_fermi_derivative_kernel!(nSt, μ, kT, energy, fermiD)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if idx <= nSt
#         ratio = (energy[idx] - μ)/kT
#         if abs(ratio) > 50.0
#             fermiD[idx] = 0.0
#         else
#             fermiD[idx] = -1.0 / (2.0*kT*(cosh(ratio) + 1.0))
#         end
#     end
#     return nothing
# end

# function comp_phi_kernel!(dim, nSt, fil, gs, fermiF, op)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if idx <= nSt
#         density = 0.0
#         for i in 1:nSt
#             density += gs[idx, i] * gs[idx, i] * fermiF[i]
#         end
#         j = (idx - 1) ÷  dim + 1
#         k = (idx - 1) %  dim + 1
#         ## (mod(j + k, 2) - 0.5) * 2.0 computes (-1)^(j+k)
#         op[idx] = (density - fil) * (mod(j + k, 2) - 0.5) * 2.0
#     end
#     return nothing
# end

function comp_density_kernel!(nSt, gs, fermiF, nn)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        density = 0.0
        for i in 1:nSt
            density += gs[idx, i] * gs[idx, i] * fermiF[i]
        end
        nn[idx] = density
    end
    return nothing
end

function comp_density_from_phi_kernel!(dim, nSt, fil, op, nn)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        ## (mod(j + k, 2) - 0.5) * 2.0 computes (-1)^(j+k)
        nn[idx] = fil + op[idx] * (mod(j + k, 2) - 0.5) * 2.0
    end
    return nothing
end

function comp_phi_from_density_kernel!(dim, nSt, fil, op, nn)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        ## (mod(j + k, 2) - 0.5) * 2.0 computes (-1)^(j+k)
        op[idx] = (nn[idx] - fil) * (mod(j + k, 2) - 0.5) * 2.0
    end
    return nothing
end

function comp_perturbation_kernel!(nSt, vals, vecs, V, E)
    jdx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if jdx <= nSt
        for mdx in 1:nSt
            for ndx in 1:nSt
                if mdx == ndx
                    continue
                else
                    pre = (vecs[jdx,mdx]*vecs[jdx,ndx])/(vals[mdx]-vals[ndx])
                    for ldx in 1:nSt
                        V[jdx, ldx, mdx] += pre * vecs[ldx, ndx]
                    end
               end
            end
            E[jdx, mdx] = vecs[jdx,mdx] * vecs[jdx,mdx]
        end
    end
    return nothing
end

function comp_Jmn_kernel!(nSt, μ, kT, vecs, vals, V, E, fermiF, fermiD, Jmn)
    jdx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if jdx <= nSt
        for idx in 1:nSt
            for kdx in 1:nSt
                Jmn[idx, jdx] += 2.0 * V[jdx, idx, kdx] * vecs[idx, kdx] * fermiF[kdx] + vecs[idx, kdx] * vecs[idx, kdx] * fermiD[kdx] * E[jdx, kdx]
            end
        end
    end
    return nothing
end

function comp_Lagrange_kernel!(nSt, vnn, vList, potential, nn, nnL, nnR, nnB, nnT, lnn)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        lnn[idx] = potential[idx] - vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nnT[idx]] + nn[nnB[idx]]) - vList[idx]
    end
    return nothing
end

function initialize_random_density(nSt, tnn, kT, μ)

    hmn   = CUDA.zeros(Float64, (nSt, nSt))
    nn    = CUDA.zeros(Float64, nSt)
    fermiF= CUDA.zeros(Float64, nSt)

    @cuda threads = nT blocks = nB init_hmn_kernel!(nSt, tnn, hmn, CUDA.rand(Float64,nSt), nnR, nnT)
    vals, vecs = eigen(hmn)
    # println(vals)
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
    @cuda threads = nT blocks = nB comp_density_kernel!(nSt, vecs, fermiF, nn)

    return nn
end

function computeNN_kernel!(dim, nSt, nnL, nnR, nnB, nnT)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        jdx = (j - 1) * dim
        bdx = mod(j - 2, dim) * dim
        tdx = mod(j    , dim) * dim
        ldx = mod(k - 2, dim) + 1
        rdx = mod(k    , dim) + 1
        nnL[idx] = jdx + ldx
        nnR[idx] = jdx + rdx
        nnB[idx] = bdx + k
        nnT[idx] = tdx + k
    end
    return nothing
end

function computeNN_cuda(nT, nB, dim, nSt)
    nnL = CUDA.zeros(Int64, nSt)
    nnR = CUDA.zeros(Int64, nSt)
    nnT = CUDA.zeros(Int64, nSt)
    nnB = CUDA.zeros(Int64, nSt)
    @cuda threads = nT blocks = nB (
        computeNN_kernel!(dim, nSt, nnL, nnR, nnB, nnT))
    return nnL,nnR,nnB,nnT
end


function selfConsistent(dim, nSt, kT, fil, tnn, vnn, W)

    hmn   = CUDA.zeros(Float64, (nSt, nSt))
    nn    = CUDA.zeros(Float64, nSt)
    fermiF= CUDA.zeros(Float64, nSt)


    Δ = fil
    vList = W * (CUDA.rand(Float64, nSt) - 0.5 * CUDA.ones(Float64, nSt))
    @cuda threads = nT blocks = nB init_CDW_kernel!(nSt, dim, fil, Δ, nn)
    @cuda threads = nT blocks = nB init_hmn_self_consistent_kernel!(nSt, tnn, hmn, vnn, vList, nn, nnL, nnR, nnT, nnB)
    valsInit, vecsInit = eigen(hmn)
    ## define the chemical potential around the filling level
    CUDA.@allowscalar μ = (valsInit[Int64(nSt * fil)] + valsInit[Int64(nSt * fil) + 1]) * 0.5
    ## compute the fermi factor
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, valsInit, fermiF)
    @cuda threads = nT blocks = nB comp_density_kernel!(nSt, vecsInit, fermiF, nn)
    for itr in 1:1000
        @cuda threads = nT blocks = nB updt_hmn_self_consistent_kernel!(nSt, hmn, vnn, vList, nn, nnL, nnR, nnT, nnB)
        vals, vecs = eigen(hmn)
        CUDA.@allowscalar μ = (vals[Int64(nSt * fil)] + vals[Int64(nSt * fil) + 1]) * 0.5
        @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
        @cuda threads = nT blocks = nB comp_density_kernel!(nSt, vecs, fermiF, nn)
        Δnew = reduce(+,map(x->abs(x-0.5),nn))/nSt
        if abs(Δnew-Δ) <1E-10
            Δ = Δnew
            println("Δ converged at $(Δ), μ: $(μ)")
            break
        else
            Δ = Δnew
        end
    end
    return μ,Δ,vList
end

function find_potential(dim, nSt, fil, kT, tnn, vList, stp, target, μ, testField=CUDA.rand(Float64,nSt))

    hmn   = CUDA.zeros(Float64, (nSt, nSt))
    op    = CUDA.zeros(Float64, nSt)
    nn    = CUDA.zeros(Float64, nSt)
    fermiF= CUDA.zeros(Float64, nSt)

    ## initialize the system with random onsite potential
    # testField = CUDA.zeros(Float64, nSt)
    # CUDA.rand!(testField)
    # testField = testField * 0.99
    # testField = target
    # CUDA.@allowscalar testField[1] = testField[1] + 1E-1
    @cuda threads = nT blocks = nB init_hmn_kernel!(nSt, tnn, hmn, testField, nnR, nnT)
    # testField = map(+, testField,CUDA.ones(Float64,nSt))
    # display(testField)
    valsInit, vecsInit = eigen(hmn)
    # CUDA.@allowscalar μInit = (valsInit[Int64(nSt * 0.5)] + valsInit[Int64(nSt * 0.5) + 1]) * 0.5
    # CUDA.@allowscalar μOut = (valsInit[Int64(nSt * 0.5)] + valsInit[Int64(nSt * 0.5) + 1]) * 0.5

    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, valsInit, fermiF)
    @cuda threads = nT blocks = nB comp_density_kernel!(nSt, vecsInit, fermiF, nn)
    oldDiff = norm(target-nn)
    delta  = stp * (target - nn) #gradient descent;
    # display(target - nn)
    println(oldDiff,",",norm(delta))
    for run in 1:5000
        # v = 0.01/norm(delta)
        # println(norm(delta))
        testField -= delta
        @cuda threads = nT blocks = nB updt_hmn_kernel!(nSt, hmn, testField)
        vals, vecs = eigen(hmn)
        # CUDA.@allowscalar μ = (vals[Int64(nSt * 0.5)] + vals[Int64(nSt * 0.5) + 1]) * 0.5
        @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
        @cuda threads = nT blocks = nB comp_density_kernel!(nSt, vecs, fermiF, nn)
        newDiff=norm(target-nn)
        if newDiff < 1E-9
            println("converged at ",run," run with precision ",newDiff)
            break
        end
        # if run>1 && newDiff > oldDiff
        #     stp /= 10.0
        #     println(stp)
        # elseif abs(oldDiff - newDiff) < 1E-4
        #     stp *= 10.0
        #     println(stp)
        # end
        oldDiff=newDiff
        delta  = stp * (target - nn)

        # CUDA.@allowscalar μOut = (vals[Int64(nSt * 0.5)] + vals[Int64(nSt * 0.5) + 1]) * 0.5
    end
    lnn = CUDA.zeros(Float64, nSt)
    vals, vecs = eigen(hmn)
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
    @cuda threads = nT blocks = nB comp_Lagrange_kernel!(nSt, vnn, vList, testField, nn, nnL, nnR, nnB, nnT, lnn)

    return testField,nn,lnn,vals
end

dim   = 64
nSt   = dim * dim
fil   = 0.5
kT    = 2e-1
tnn   = 2.0
vnn   = 1.0
W     = 0.0  ## disorder strength:  vnn + [-W/2,W/2]
## step for convergence of single-Slater-determinant state
stp   = 5e-0
## step for evolution of order parameter dynamics
dt    = 5e-3 #time step; 

nT    = 512
nB    = cld(nSt, nT)

CUDA.seed!(parse(Int, ARGS[1]))

nnL,nnR,nnB,nnT = computeNN_cuda(nT, nB, dim, nSt)

# mask   = readdlm("mask.txt", Int64)
# input  = CuArray(vec((mask .-0.5) * 2.0)) * 0.2949348978881737

# input = CuArray(vec((mask .-0.5) * 2.0)) * 0.3
# input = CuArray(vec((mask .-0.5) * 2.0)) * 0.3453282000310052
# input = CuArray(vec((mask .-0.5) * 2.0)) * 0.2949348978881737

μ,Δ,vList = selfConsistent(dim, nSt, kT, fil, tnn, vnn, W)
println("Converged chemical potential: ", μ, " converged order parameter: ", Δ)
## convert
# input   = CuArray(readdlm("mask.txt", Float64))
# target = CUDA.zeros(Float64, nSt)
# let target = CUDA.zeros(Float64, nSt), vConfig, nn, dnn, op, output
    # @cuda threads = nT blocks = nB comp_density_from_phi_kernel!(dim, nSt, fil, input, target)

    target = initialize_random_density(nSt, tnn, kT, μ) #initialize random configuratoin; 
    vConfig,nn,lnn,energy = find_potential(dim, nSt, fil, kT, tnn, vList, stp, target, μ) # find the force of the potential

    # op    = CUDA.zeros(Float64, nSt)
    # @cuda threads = nT blocks = nB comp_phi_from_density_kernel!(dim, nSt, reduce(+,nn)/nSt, op, nn)
    output = nn
    force  = lnn
    # dos    = energy

    for run = 1:10000
        # if run > 50
        #     global stp = 5E-0
        # end

        global target += (lnn +  (CUDA.rand(Float64, nSt) - 0.5 * CUDA.ones(Float64, nSt)) * 2.0 * Δ * 0.01 ) * dt  #traget is the updated density; 
        # global target +=  lnn * dt
        global vConfig,nn,lnn,energy = find_potential(dim, nSt, fil, kT, tnn, vList, stp, target, μ, vConfig) # find the updated potential

        # opN           = CUDA.zeros(Float64, nSt)
        # chargeDensity = reduce(+,nn)/nSt
        # @cuda threads = nT blocks = nB comp_phi_from_density_kernel!(dim, nSt, chargeDensity, opN, nn)

        if run % 10 ==0  # save enery 10 steps
            global output = [output;nn]
            global force  = [force; lnn]
            # global dos    = [dos;energy]
            println("step $(run) done")
            if run % 1000 == 0
                writedlm("density-$(ARGS[1]).txt",Array(output))
                writedlm("force-$(ARGS[1]).txt",Array(force))
            end
        end
        # dnn    = update_nn(dim, nSt, μ, kT, tnn, vnn, vConfig, nn)
    end

    println("charge density:",reduce(+,nn)/nSt)
    writedlm("density-$(ARGS[1]).txt",Array(output))
    writedlm("force-$(ARGS[1]).txt",Array(force))
    # writedlm("dos-$(ARGS[1]).txt",Array(dos))
    # writedlm("force.txt",Array(lnn))
# end
