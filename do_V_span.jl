using OrdinaryDiffEq
using ProgressBars
using Distributions

const m87 = 1.44316060e-25
const k_B = 1.38064852e-23
const u = sqrt(2 * k_B * T / m87)
const l = u * (1 / abs(Gamma))

function bresenham(x1::Int32, y1::Int32, x2::Int32, y2::Int32)::Array{Array{Int32,1},1}
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
    for x in x1:x2
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
        reverse!(points)
    end
    return points
end

function choose_points()
    """
    Randomly chooses 2 points at the boundary of the grid.
    """
    edges = Array{Tuple{Int32,Int32},1}(undef, 4 * N_grid)
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
    """
    Draws a random longitudinal velocity given a total velocity v.
    Uses Maxwell-Blotzmann distribution.
    """
    vz = abs(2 * v)
    while abs(vz) > abs(v)
        vz = randn() * sqrt(k_B * T / m87)
    end
    return vz
end

@inbounds begin
    @fastmath begin
        function f!(dy::Array{ComplexF64,1}, x::Array{ComplexF64,1},
            p::Array{ComplexF64,1}, t::Float64)
            """
            RHS of the Maxwell-Bloch equations. This needs to be *fast*.
            Calculation of the RHS is done in place. The calculation is carried out in an unrolled 
            manner to avoid allocating a 8x8 array each call (unefficient for such a small system).
            """
            r_sq = (p[4] + p[2] * p[1] * t - p[13])^2 + (p[5] + p[3] * p[1] * t - p[13])^2
            Om23 = p[8] * exp(-r_sq / (2.0 * p[12] * p[12]))
            Om13 = p[7] * exp(-r_sq / (2.0 * p[12] * p[12]))
            if t >= abs(p[14]) && abs(p[15]) == 0
                dy[1] = x[2]
                dy[2] = x[1]
                dy[3] = x[4]
                dy[4] = x[3]
                dy[5] = (im * Om23 / 2.0) * x[1] + im * Om23 * x[2] + (im * Om13 / 2.0) * x[4] - p[11] * x[7] - im * Om23 / 2.0
                dy[6] = -im * conj(Om13) * x[1] - im * (conj(Om13) / 2.0) * x[2] - (im * conj(Om23) / 2.0) * x[4] - conj(p[10]) * x[6] + im * conj(Om13) / 2.0
                dy[7] = im * Om13 * x[1] + (im * Om13 / 2.0) * x[2] + (im * Om23 / 2.0) * x[3] - p[10] * x[5] - im * Om13 / 2.0
                dy[8] = (-im * conj(Om23) / 2.0) * x[1] - im * conj(Om23) * x[2] - (im * conj(Om13) / 2.0) * x[3] - conj(p[11]) * x[8] + im * conj(Om23) / 2.0
                p[15] += 1
            else
                dy[1] = (-p[6] / 2.0) * x[1] - (p[6] / 2.0) * x[2] + (im * conj(Om13) / 2.0) * x[5] - (im * Om13 / 2) * x[6] + p[6] / 2.0
                dy[2] = (-p[6] / 2.0) * x[1] - (p[6] / 2.0) * x[2] + (im * conj(Om23) / 2.0) * x[7] - (im * Om23 / 2) * x[8] + p[6] / 2.0
                dy[3] = -p[9] * x[3] + (im * conj(Om23) / 2.0) * x[5] - (im * Om13 / 2.0) * x[8]
                dy[4] = -conj(p[9]) * x[4] - (im * Om23 / 2.0) * x[6] + (im * conj(Om13) / 2.0) * x[7]
                dy[5] = im * Om13 * x[1] + (im * Om13 / 2.0) * x[2] + (im * Om23 / 2.0) * x[3] - p[10] * x[5] - im * Om13 / 2.0
                dy[6] = -im * conj(Om13) * x[1] - im * (conj(Om13) / 2.0) * x[2] - (im * conj(Om23) / 2.0) * x[4] - conj(p[10]) * x[6] + im * conj(Om13) / 2.0
                dy[7] = (im * Om23 / 2.0) * x[1] + im * Om23 * x[2] + (im * Om13 / 2.0) * x[4] - p[11] * x[7] - im * Om23 / 2.0
                dy[8] = (-im * conj(Om23) / 2.0) * x[1] - im * conj(Om23) * x[2] - (im * conj(Om13) / 2.0) * x[3] - conj(p[11]) * x[8] + im * conj(Om23) / 2.0
            end
        end
    end
end


@inbounds begin
    @fastmath begin
        function f_jac!(J::Array{ComplexF64,2}, x::Array{ComplexF64,1},
            p::Array{ComplexF64,1}, t::Float64)
            """
            Jacobian of the Maxwell-Bloch system
            """
            r_sq = (p[4] + p[2] * p[1] * t - p[13])^2 + (p[5] + p[3] * p[1] * t - p[13])^2
            Om23 = p[8] * exp(-r_sq / (2 * p[12] * p[12]))
            Om13 = p[7] * exp(-r_sq / (2 * p[12] * p[12]))
            J[1, 1] = -p[6] / 2.0
            J[1, 2] = -p[6] / 2.0
            J[1, 3] = 0.0 * im
            J[1, 4] = 0.0 * im
            J[1, 5] = im * conj(Om13) / 2.0
            J[1, 6] = -im * Om13 / 2.0
            J[1, 7] = 0.0 * im
            J[1, 8] = 0.0 * im
            J[2, 1] = (-p[6] / 2.0)
            J[2, 2] = -(p[6] / 2.0)
            J[2, 3] = 0.0 * im
            J[2, 4] = 0.0 * im
            J[2, 5] = 0.0 * im
            J[2, 6] = 0.0 * im
            J[2, 7] = im * conj(Om23) / 2.0
            J[2, 8] = -im * Om23 / 2.0
            J[3, 1] = 0.0 * im
            J[3, 2] = 0.0 * im
            J[3, 3] = -p[9]
            J[3, 4] = 0.0 * im
            J[3, 5] = im * conj(Om23) / 2.0
            J[3, 6] = 0.0 * im
            J[3, 7] = 0.0 * im
            J[3, 8] = -im * Om13 / 2.0
            J[4, 1] = 0.0 * im
            J[4, 2] = 0.0 * im
            J[4, 3] = 0.0 * im
            J[4, 4] = -conj(p[9])
            J[4, 5] = 0.0 * im
            J[4, 6] = -im * Om23 / 2.0
            J[4, 7] = im * conj(Om13) / 2.0
            J[4, 8] = 0.0 * im
            J[5, 1] = im * Om13
            J[5, 2] = im * Om13 / 2.0
            J[5, 3] = im * Om23 / 2.0
            J[5, 4] = 0.0 * im
            J[5, 5] = -p[10]
            J[5, 6] = 0.0 * im
            J[5, 7] = 0.0 * im
            J[5, 8] = 0.0 * im
            J[6, 1] = -im * conj(Om13)
            J[6, 2] = -im * (conj(Om13) / 2.0)
            J[6, 3] = 0.0 * im
            J[6, 4] = -(im * conj(Om23) / 2.0)
            J[6, 5] = 0.0 * im
            J[6, 6] = -conj(p[10])
            J[6, 7] = 0.0 * im
            J[6, 8] = 0.0 * im
            J[7, 1] = (im * Om23 / 2.0)
            J[7, 2] = im * Om23
            J[7, 3] = 0.0 * im
            J[7, 4] = im * Om13 / 2.0
            J[7, 5] = 0.0 * im
            J[7, 6] = 0.0 * im
            J[7, 7] = -p[11]
            J[7, 8] = 0.0 * im
            J[8, 1] = -im * conj(Om23) / 2.0
            J[8, 2] = -im * conj(Om23)
            J[8, 3] = -im * conj(Om13) / 2.0
            J[8, 4] = 0.0 * im
            J[8, 5] = 0.0 * im
            J[8, 6] = 0.0 * im
            J[8, 7] = 0.0 * im
            J[8, 8] = -conj(p[11])
        end
    end
end


# Instantiates the various accumulator grids : note that everything should be preallocated
# with *fixed* size, immutable arrays for speed. 
grid_13 = zeros(ComplexF64, (N_grid, N_grid))
grid_23 = zeros(ComplexF64, (N_grid, N_grid))
counter_grid = zeros(Int32, (N_grid, N_grid))
counter_grid_total = zeros(Int32, (N_grid, N_grid))
grid_weighted_13 = zeros(ComplexF64, (N_grid, N_grid))
grid_weighted_23 = zeros(ComplexF64, (N_grid, N_grid))
Vs = collect(LinRange{Float64}(v0, v1, N_v))
pv = sqrt(2.0 / pi) * ((m87 / (k_B * T))^(3.0 / 2.0)) .* Vs .^ 2.0 .* exp.(-m87 * Vs .^ 2.0 / (2.0 * k_B * T))
v_perps = zeros(Float64, N_real)
global coords = Array{Tuple{Int32,Int32,Int32,Int32}}(undef, N_real)
# Loop over velocity classes
for (index_v, v) in ProgressBar(enumerate(Vs))
    global paths = [[] for i = 1:N_real]
    global tpaths = [[] for i = 1:N_real]
    # Uncomment to experiment with collisions
    # global t_colls = rand(Exponential(l / v), N_real)
    global t_colls = -1.0 * ones(Float64, N_real)
    # Choose points in advance to save only the relevant points during solving
    Threads.@threads for i = 1:N_real
        iinit, jinit, ifinal, jfinal = choose_points()
        coords[i] = (iinit, jinit, ifinal, jfinal)
        paths[i] = bresenham(jinit, iinit, jfinal, ifinal)
        v_perps[i] = sqrt(v^2.0 - draw_vz(v)^2.0)
        tpaths[i] = sort(Float64[hypot(coord[2] - iinit, coord[1] - jinit) * abs(window) / (v_perps[i] * N_grid) for coord in paths[i]])
    end
    # Instantiates a new problem
    function prob_func(prob, i, repeat)
        iinit, jinit, ifinal, jfinal = coords[i]
        v_perp = v_perps[i]
        vz = sqrt(v^2.0 - v_perp^2.0)
        xinit = jinit * window / N_grid
        yinit = iinit * window / N_grid
        xfinal = jfinal * window / N_grid
        yfinal = ifinal * window / N_grid
        if v_perp != 0
            u0 = xfinal - xinit
            u1 = yfinal - yinit
            norm = hypot(u0, u1)
            u0 /= norm
            u1 /= norm
        else
            u0 = u1 = 0
        end
        new_p = ComplexF64[v_perp+0.0*im, u0+0.0*im, u1+0.0*im,
            xinit+0.0*im, yinit+0.0*im,
            Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde-im*k*vz,
            gamma32tilde-im*k*vz, waist, r0, t_colls[i], 1.0*im]
        # Uncomment to experiment with collisions
        # new_p = ComplexF64[v_perp+0.0*im, u0+0.0*im, u1+0.0*im,
        #     xinit+0.0*im, yinit+0.0*im,
        #     Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde-im*k*vz,
        #     gamma32tilde-im*k*vz, waist, r0, t_colls[i], 0.0*im]
        remake(prob, p=new_p, tspan=(0.0, maximum(tpaths[i])), saveat=tpaths[i])

    end

    p = ComplexF64[1.0+0.0*im, 1.0+0.0*im, 1.0+0.0*im, 1.0+0.0*im, 1.0+0.0*im,
        Gamma, Omega13,
        Omega23, gamma21tilde,
        gamma31tilde-im*k*0.0,
        gamma32tilde-im*k*0.0, waist, r0, 1.0, 0]
    tspan = (0.0, 1.0)
    tsave = collect(LinRange{Float64}(tspan[1], tspan[2], 2))
    prob = ODEProblem{true}(f!, x0, tspan, p, jac=f_jac!, saveat=tsave)
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    # for best precision
    # alg = KenCarp58(autodiff=false)
    alg = TRBDF2(autodiff=false)
    atol = 1e-10
    rtol = 1e-8
    # run once on small system to try and speed up compile time
    if index_v == 1
        sol = solve(ensembleprob, alg, EnsembleThreads(),
            trajectories=2, save_idxs=[5, 7], abstol=atol, reltol=rtol,
            maxiters=Int(1e8), dt=1e-14, dtmin=1e-14, dtmax=1e-5)
    end
    # Solve main problem in parallel over threads. 
    sol = solve(ensembleprob, alg, EnsembleThreads(),
        trajectories=N_real, save_idxs=[5, 7], abstol=atol, reltol=rtol,
        maxiters=Int(1e8), dt=1e-14, dtmin=1e-16, dtmax=1e-6)
    global grid_13 .= zeros(ComplexF64, (N_grid, N_grid))
    global grid_23 .= zeros(ComplexF64, (N_grid, N_grid))
    global counter_grid .= zeros(Int32, (N_grid, N_grid))
    @inbounds begin
        Threads.@threads for i = 1:N_real
            for (j, coord) in enumerate(paths[i])
                grid_13[coord[2], coord[1]] += sol[i].u[j][1]
                grid_23[coord[2], coord[1]] += sol[i].u[j][2]
                counter_grid[coord[2], coord[1]] += 1
            end
        end
    end
    # Averaging over realizations and velocity probability
    grid_weighted_13 .+= (grid_13 ./ counter_grid) * pv[index_v] * abs(Vs[2] - Vs[1])
    grid_weighted_23 .+= (grid_23 ./ counter_grid) * pv[index_v] * abs(Vs[2] - Vs[1])
    counter_grid_total .+= counter_grid

end

[grid_weighted_13, grid_weighted_23, counter_grid_total]