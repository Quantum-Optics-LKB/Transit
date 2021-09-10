using OrdinaryDiffEq
# using DiffEqGPU, OrdinaryDiffEq
# # for GPU try only with cuArrays in Float64 precision
using PyPlot

t_tot0 = time()
# global variables to be set from python side through Julia "Main" namespace
const N_real = 5000
const N_grid = 256
const N_t = 1000
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
const r0 = 1.5*waist
const window = 3*waist
const x0 = ComplexF64[5/8 + 0*im, 3/8+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im]
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
function bresenham(x1::Int32, y1::Int32, x2::Int32, y2::Int32)::Array{Array{Int32, 1}, 1}

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

function draw_vz(v::Float64)::Float64
    vz = abs(2*v)
    while abs(vz) > abs(v)
        vz = randn()*sqrt(k_B*T/m87)
    end
    return vz
end

function f!(dy::Array{ComplexF64, 1}, x::Array{ComplexF64, 1},
            p::Array{ComplexF64, 1}, t::Float64)
    r_sq = (p[4]+p[2]*p[1]*t - p[13])*(p[4]+p[2]*p[1]*t - p[13]) + (p[5]+p[3]*p[1]*t - p[13])*(p[5]+p[3]*p[1]*t - p[13])
    Om23 = p[8] * exp(-r_sq/(2.0*p[12]*p[12]))
    Om13 = p[7] * exp(-r_sq/(2.0*p[12]*p[12]))
    dy[1] = (-p[6]/2)*x[1]-(p[6]/2)*x[2]+(im*conj(Om13)/2)*x[5]-(im*Om13/2)*x[6]+p[6]/2
    dy[2] = (-p[6]/2)*x[1]-(p[6]/2)*x[2]+(im*conj(Om23)/2)*x[7]-(im*Om23/2)*x[8]+p[6]/2
    dy[3] = -p[9]*x[3]+(im*conj(Om23)/2)*x[5]-(im*Om13/2)*x[8]
    dy[4] = -conj(p[9])*x[4] - (im*Om23/2)*x[6] + (im*conj(Om13)/2)*x[7]
    dy[5] = im*Om13*x[1] + (im*Om13/2)*x[2] + (im*Om23/2)*x[3] - p[10]*x[5]-im*Om13/2
    dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2)*x[2]-(im*conj(Om23)/2)*x[4]-conj(p[10])*x[6]+im*conj(Om13)/2
    dy[7] = (im*Om23/2)*x[1]+im*Om23*x[2]+(im*Om13/2)*x[4]-p[11]*x[7] - im*Om23/2
    dy[8] = (-im*conj(Om23)/2)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2)*x[3]-conj(p[11])*x[8]+im*conj(Om23)/2
end

function f_jac!(J::Array{ComplexF64, 2}, x::Array{ComplexF64, 1},
            p::Array{ComplexF64, 1}, t::Float64)
    r_sq = (p[4]+p[2]*p[1]*t - p[13])^2 + (p[5]+p[3]*p[1]*t - p[13])^2
    Om23 = p[8] * exp(-r_sq/(2*p[12]*p[12]))
    Om13 = p[7] * exp(-r_sq/(2*p[12]*p[12]))
    J[1, 1] = (-p[6]/2.0)
    J[1, 2] = -(p[6]/2.0)
    J[1, 3] = 0.0 * im
    J[1, 4] = 0.0 * im
    J[1, 5] = (im*conj(Om13)/2.0)
    J[1, 6] = -(im*Om13/2.0)
    J[1, 7] = 0.0 * im
    J[1, 8] = 0.0 * im
    J[2, 1] = (-p[6]/2.0)
    J[2, 2] = -(p[6]/2.0)
    J[2, 3] = 0.0 * im
    J[2, 4] = 0.0 * im
    J[2, 5] = 0.0 * im
    J[2, 6] = 0.0 * im
    J[2, 7] = (im*conj(Om23)/2.0)
    J[2, 8] = -(im*Om23/2.0)
    J[3 ,1] = 0.0 * im
    J[3 ,2] = 0.0 * im
    J[3, 3] = -p[9]
    J[3 ,4] = 0.0 * im
    J[3, 5] = (im*conj(Om23)/2.0)
    J[3 ,6] = 0.0 * im
    J[3 ,7] = 0.0 * im
    J[3, 8] = -(im*Om13/2.0)
    J[4, 1] = 0.0 * im
    J[4, 2] = 0.0 * im
    J[4, 3] = 0.0 * im
    J[4, 4] = -conj(p[9])
    J[4, 5] = 0.0 * im
    J[4, 6] = -(im*Om23/2.0)
    J[4, 7] = (im*conj(Om13)/2.0)
    J[4, 8] = 0.0 * im
    J[5, 1] = im*Om13
    J[5, 2] = (im*Om13/2.0)
    J[5, 3] = (im*Om23/2.0)
    J[5, 4] = 0.0 * im
    J[5, 5] = - p[10]
    J[5, 6] = 0.0 * im
    J[5, 7] = 0.0 * im
    J[5, 8] = 0.0 * im
    J[6, 1] = -im*conj(Om13)
    J[6, 2] = -im*(conj(Om13)/2.0)
    J[6, 3] = 0.0 * im
    J[6, 4] = -(im*conj(Om23)/2.0)
    J[6, 5] = 0.0 * im
    J[6, 6] = -conj(p[10])
    J[6, 7] = 0.0 * im
    J[6, 8] = 0.0 * im
    J[7, 1] = (im*Om23/2.0)
    J[7, 2] = +im*Om23*x[2]
    J[7, 3] = 0.0 * im
    J[7, 4] = +(im*Om13/2.0)
    J[7, 5] = 0.0 * im
    J[7, 6] = 0.0 * im
    J[7, 7] = -p[11]
    J[7, 8] = 0.0 * im
    J[8, 1] = (-im*conj(Om23)/2.0)
    J[8, 2] = -im*conj(Om23)
    J[8, 3] = -(im*conj(Om13)/2.0)
    J[8, 4] = 0.0 * im
    J[8, 5] = 0.0 * im
    J[8, 6] = 0.0 * im
    J[8, 7] = 0.0 * im
    J[8, 8] = -conj(p[11])
end

# p0 = ComplexF64[1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im,
#      Gamma, Omega13,
#      Omega23, gamma21tilde,
#      gamma31tilde - im*k*0.0,
#      gamma32tilde - im*k*0.0, waist, r0]
# J = zeros(ComplexF64, (8, 8))
# dy = similar(x0)
# @code_warntype f!(dy, x0, p0, 1.0)
# @code_warntype f_jac!(J, x0, p0, 1.0)
# @time f!(dy, x0, p0, 1.0)
# @time f_jac!(J, x0, p0, 1.0)


#TODO directly put an accumulator array of size (N_grid, N_grid)
# grids = zeros(ComplexF64, (N_v, N_grid, N_grid))
# counter_grids = zeros(Int32, (N_v, N_grid, N_grid))
# define empty grid to accumulate the trajectories of the atoms and count them
grid = zeros(ComplexF64, (N_grid, N_grid))
counter_grid = zeros(Int32, (N_grid, N_grid))
counter_grid_total = zeros(Int32, (N_grid, N_grid))
grid_weighted = zeros(ComplexF64, (N_grid, N_grid))
normalized = zeros(Float64, (N_grid, N_grid))
Vs = collect(LinRange{Float64}(v0, v1, N_v))
pv = sqrt(2/pi)*((m87/(k_B*T))^(3/2)).*Vs.^2 .*exp.(-m87*Vs.^2/(2*k_B*T))
Xs = zeros(ComplexF64, (N_t, N_real))
Ts = zeros(Float64, (N_t, N_real))
v_perps = zeros(Float64, N_real)
paths = [[] for i=1:N_real]

for (index_v, v) in enumerate(Vs)
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
        new_p = ComplexF64[v_perp + 0*im, u0 + 0*im, u1 + 0*im, xinit + 0*im, yinit + 0*im,
                 Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde - im*k*vz,
                 gamma32tilde - im*k*vz, waist, r0]
        new_tspan = (0.0, Float64(new_tfinal))
        tsave = collect(LinRange{Float64}(0.0, Float64(new_tfinal), N_t))
        remake(prob, p=new_p, tspan=new_tspan, saveat=tsave)
    end
    # instantiate a problem
    p = ComplexF64[1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im,
         Gamma, Omega13,
         Omega23, gamma21tilde,
         gamma31tilde - im*k*0.0,
         gamma32tilde - im*k*0.0, waist, r0]
    tspan = (0.0, 1.0)
    tsave = collect(LinRange{Float64}(0.0, 1.0, N_t))
    prob = ODEProblem{true}(f!, x0, tspan, p, jac=f_jac!, saveat=tsave)
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    # run once on small system to try and speed up compile time
    if index_v==1
        t0 = time()
        sol = solve(ensembleprob, Rodas5(autodiff=false), EnsembleThreads(),
                          trajectories=2, save_idxs=7,
                          abstol=1e-6, reltol=1e-3)
        t1 = time()
        println("Time spent to solve first time : $(t1-t0) s")

    end
    t0 = time()
    sol = solve(ensembleprob, Rodas5(autodiff=false), EnsembleThreads(),
                      trajectories=N_real, save_idxs=7,
                      abstol=1e-6, reltol=1e-3)
    t1 = time()
    println("Time spent to solve realizations $(index_v)/$(N_v) : $(t1-t0) s")
    # @time sol = solve(ensembleprob, BS3(), EnsembleGPUArray(), trajectories=N_real)
    t0 = time()
    Threads.@threads for i in 1:N_real
        # what I would like
        Xs[:, i] .= sol[i].u
        Ts[:, i] .= sol[i].t
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
    # Threads.@threads for i = 1:N_real
    #     for j = 1:N_t
    #         Xs[j, i] = xs[i][j]
    #         Ts[j, i] = ts[i][j]
    #     end
    # end
    t1 = time()
    println("Time spent reformatting the output : $(t1-t0) s")
    function treat_coord!(i::Int64, coord::Array{Int32, 1}, iinit::Int32,
                          jinit::Int32, grid::Array{ComplexF64, 2},
                          counter_grid::Array{Int32, 2})
        if coord[1] > N_grid
            coord[1] = N_grid
        end
        if coord[2] > N_grid
            coord[2] = N_grid
        end
        if coord[1] < 1
            coord[1] = 1
        end
        if coord[2] < 1
            coord[2] = 1
        end
        tpath = hypot(coord[2]-iinit, coord[1]-jinit)*abs(window)/(v_perps[i]*N_grid)
        grid[coord[2], coord[1]] += Xs[argmin(abs.(Ts[:, i] .- tpath)), i]
        counter_grid[coord[2], coord[1]] += 1
    end

    t0 = time()
    global grid .= zeros(ComplexF64, (N_grid, N_grid))
    global counter_grid .= zeros(Int32, (N_grid, N_grid))
    Threads.@threads for i = 1:N_real
        local iinit = paths[i][1][1]
        local jinit = paths[i][1][2]
        m_func(coord) = treat_coord!(i, coord, iinit, jinit, grid, counter_grid)
        map(m_func, paths[i])
    end
    t1 = time()
    println("Time spent to treat realizations : $(t1-t0) s")
    grid_weighted .+= (grid./counter_grid) * pv[index_v]
    counter_grid_total .+= counter_grid
end
# weighted_grid = real.(grids).*pv
# pol = sum(weighted_grid, dims=1)
# total_counts = sum(counter_grids, dims=1)
pol = real.(grid_weighted)
println("Total time elapsed : $(time()-t_tot0) s")
fig, ax = subplots(1, 2)
ax[1].imshow(pol)
ax[2].imshow(counter_grid_total)
ax[1].set_title("Polarization")
ax[2].set_title("Counts")
show()
