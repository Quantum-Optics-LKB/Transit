using DifferentialEquations
# using DiffEqGPU, OrdinaryDiffEq
# # for GPU try only with cuArrays in Float32 precision
using PyPlot

# global variables to be set from python side through Julia "Main" namespace
const N_real = 100
const N_grid = 256
const N_v = 20
const T = 150+273
const m87 = 1.44316060e-25
const k_B = 1.38064852e-23
const Gamma = 38107518.88804419 + 0*im
const Omega13 = 135886166.68478012 + 0*im
const Omega23 = 0.021626955125691984e9 + 0*im
const gamma21tilde = 321033.05667335045+42939288389.26529*im
const gamma31tilde = 46564466.011063695+61788844310.80405*im
const gamma32tilde = 46564466.011063695+18849555921.538757*im
const waist = 1.0e-3 + 0*im
const r0 = waist/1.5
const window = 3*waist
const x0 = ComplexF32[5/8 + 0*im, 3/8+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im]
const k = 2*pi/780.241e-9
const v0 = 40.0
const v1 = 800.0
# function for line generation
"""
Bresenham's Line Algorithm
Produces a list of tuples from start and end

>>> points1 = get_line((0, 0), (3, 4))
>>> points2 = get_line((3, 4), (0, 0))
>>> assert(set(points1) == set(points2))
>>> print points1
[(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
>>> print points2
[(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
"""
function bresenham(x1::Int32, y1::Int32, x2::Int32, y2::Int32)

    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    end
    # Swap start and end points if necessary and store swap state
    swapped = false
    if x1 > x2
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = true
    end
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = dx / 2
    if y1 < y2
        ystep = 1
    else
        ystep = -1
    end

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in x1:x2+1
        if is_steep
            coord = [y, x]
        else
            coord = [x, y]
        end
        append!(points, [coord])
        error -= abs(dy)
        if error < 0
            y += ystep
            error += dx
        end
    end
    # Reverse the list if the coordinates were swapped
    if swapped
        reverse(points)
    end
    return points
end

function choose_points()
    edges = Array{Tuple{Int32, Int32}, 1}(undef, 4*N_grid)
    for i in 1:N_grid
        edges[i] = (1, i)
        edges[i+N_grid] = (N_grid, i)
        edges[i+2*N_grid] = (i, 1)
        edges[i+3*N_grid] = (i, N_grid)
    end
    iinit, jinit = edges[rand(1:length(edges), 1)][1]
    ifinal, jfinal = iinit, jinit
    cdtn = (ifinal == iinit) || (jfinal == jinit)
    while cdtn
        ifinal, jfinal = edges[rand(1:length(edges), 1)][1]
        cdtn = (ifinal == iinit) || (jfinal == jinit)
    end
    return (iinit, jinit, ifinal, jfinal)
end

function draw_vz(v::Float32)::Float32
    vz = abs(2*v)
    while abs(vz) > abs(v)
        vz = randn()*sqrt(k_B*T/m87)
    end
    return vz
end

function f!(dy::Array{ComplexF32, 1}, x::Array{ComplexF32, 1},
            p::Array{ComplexF32, 1}, t::Float32)
    r_sq = (p[4]+p[2]*p[1]*t - p[13])^2 + (p[5]+p[3]*p[1]*t - p[13])^2
    Om23 = p[8] * exp(-r_sq/(2*p[12]*p[12]))
    Om13 = p[7] * exp(-r_sq/(2*p[12]*p[12]))
    dy[1] = (-p[6]/2)*x[1]-(p[6]/2)*x[2]+(im*conj(Om13)/2)*x[5]-(im*Om13/2)*x[6]+p[6]/2
    dy[2] = (-p[6]/2)*x[1]-(p[6]/2)*x[2]+(im*conj(Om23)/2)*x[7]-(im*Om23/2)*x[8]+p[6]/2
    dy[3] = -p[9]*x[3]+(im*conj(Om23)/2)*x[5]-(im*Om13/2)*x[8]
    dy[4] = -conj(p[9])*x[4] - (im*Om23/2)*x[6] + (im*conj(Om13)/2)*x[7]
    dy[5] = im*Om13*x[1] + (im*Om13/2)*x[2] + (im*Om23/2)*x[3] - p[10]*x[5]-im*Om13/2
    dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2)*x[2]-(im*conj(Om23)/2)*x[4]-conj(p[10])*x[6]+im*conj(Om13)/2
    dy[7] = (im*Om23/2)*x[1]+im*Om23*x[2]+(im*Om13/2)*x[4]-p[11]*x[7] - im*Om23/2
    dy[8] = (-im*conj(Om23)/2)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2)*x[3]-conj(p[11])*x[8]+im*conj(Om23)/2
end

@time begin
grids = zeros(ComplexF32, (N_v, N_grid, N_grid))
counter_grids = zeros(Int32, (N_v, N_grid, N_grid))
Vs = collect(LinRange{Float32}(v0, v1, N_v))
pv = sqrt(2/pi)*((m87/(k_B*T))^(3/2)).*Vs.^2 .*exp.(-m87*Vs.^2/(2*k_B*T))
for (index_v, v) in enumerate(Vs)
    paths = [[] for i=1:N_real]
    xs = [[] for i=1:N_real]
    ts = [[] for i=1:N_real]
    v_perps = zeros(Float32, N_real)
    function prob_func(prob, i, repeat)
        # change seed of random number generation as the solver uses multithreading
        iinit, jinit, ifinal, jfinal = choose_points()
        vz = draw_vz(v)
        v_perp = sqrt(v^2 - vz^2)
        xinit = jinit*window/N_grid
        yinit = iinit*window/N_grid
        xfinal = jfinal*window/N_grid
        yfinal = ifinal*window/N_grid
        # velocity unit vector
        if v_perp != 0
            u0 = xfinal-xinit
            u1 = yfinal-yinit
            norm = hypot(u0, u1)
            u0 /= norm
            u1 /= norm
            new_tfinal = hypot((xfinal-xinit), (yfinal-yinit))/v_perp
        else
            u0 = u1 = 0
            new_tfinal = hypot((xfinal-xinit), (yfinal-yinit))/abs(vz)
        end
        new_p = ComplexF32[v_perp + 0*im, u0 + 0*im, u1 + 0*im, xinit + 0*im, yinit + 0*im,
                 Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde - im*k*vz,
                 gamma32tilde - im*k*vz, waist, r0]
        new_tspan = (0.0f0, Float32(new_tfinal))
        tsave = collect(LinRange{Float32}(0.0f0, Float32(new_tfinal), 1000))
        remake(prob, p=new_p, tspan=new_tspan, saveat=tsave)
    end
    # instantiate a problem
    p = ComplexF32[1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im,
         Gamma, Omega13,
         Omega23, gamma21tilde,
         gamma31tilde - im*k*0.0,
         gamma32tilde - im*k*0.0, waist, r0]
    tspan = (0.0f0, 1.0f0)
    tsave = collect(LinRange{Float32}(0.0f0, 1.0f0, 1000))
    prob = ODEProblem{true}(f!, x0, tspan, p, saveat=tsave)
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    @time sol = solve(ensembleprob, Tsit5(), EnsembleThreads(),
                      trajectories=N_real, maxiters=Int(1e8))
    # @time sol = solve(ensembleprob, BS3(), EnsembleGPUArray(), trajectories=N_real)
    for i in 1:N_real
        xs[i] = sol[i].u
        ts[i] = sol[i].t
        v_perps[i] = sol[i].prob.p[1]
        local u0 = sol[i].prob.p[2]
        local u1 = sol[i].prob.p[3]
        local xinit = sol[i].prob.p[4]
        local yinit = sol[i].prob.p[5]
        local tfinal = sol[i].prob.tspan[2]
        local xfinal = xinit+u0*v_perps[i]*tfinal
        local yfinal = yinit+u1*v_perps[i]*tfinal
        local iinit = Int32(round(real(yinit)*(N_grid/window), digits=1))
        local jinit = Int32(round(real(xinit)*(N_grid/window), digits=1))
        local ifinal = Int32(round(real(yfinal)*(N_grid/window), digits=1))
        local jfinal = Int32(round(real(xfinal)*(N_grid/window), digits=1))
        local path = bresenham(jinit, iinit, jfinal, ifinal)
        paths[i] = path
    end
    # Reformat xs as a proper multidimensional Array for easier indexing
    Xs = zeros(ComplexF32, (N_real, length(ts[1]), 8))
    for i = 1:N_real
        for j = 1:length(ts[1])
            for k in 1:8
                Xs[i, j, k] = xs[i][j][k]
            end
        end
    end
    xs = nothing
    # define empty grid to accumulate the trajectories of the atoms and count them
    grid = zeros(ComplexF32, (N_grid, N_grid))
    counter_grid = zeros(Int32, (N_grid, N_grid))
    for i = 1:N_real
        local iinit = paths[i][1][1]
        local jinit = paths[i][1][2]
        for coord in paths[i]
            if coord[1] > N_grid-1
                coord[1] = N_grid-1
            end
            if coord[2] > N_grid-1
                coord[2] = N_grid-1
            end
            if coord[1] < 1
                coord[1] = 1
            end
            if coord[2] < 1
                coord[2] = 1
            end
            tpath = hypot(coord[2]-iinit, coord[1]-jinit)*window/(v_perps[i]*N_grid)
            grid[coord[2], coord[1]] += real(Xs[i, argmin(abs.([ts[i][k]-tpath for k=1:length(ts[i])])), 7])
            counter_grid[coord[2], coord[1]] += 1
        end
    end
    grids[index_v, :, :] .= grid
    counter_grids[index_v, :, :] .= counter_grid
end
end
weighted_grid = real.(grids).*pv
pol = sum(weighted_grid, dims=1)
total_counts = sum(counter_grids, dims=1)
normalized = pol./total_counts
fig, ax = subplots(1, 2)
ax[1].imshow(normalized[1, :, :])
ax[2].imshow(total_counts[1, :, :])
ax[1].set_title("Polarization")
ax[2].set_title("Counts")
show()
