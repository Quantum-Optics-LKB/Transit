using OrdinaryDiffEq
# using DiffEqGPU, OrdinaryDiffEq
# # for GPU try only with cuArrays in Float64 precision
using PyPlot
using ProgressBars

# global variables to be set from python side through Julia "Main" namespace
const N_real = 10000
const N_grid = 128
const N_v = 20
const T = 150+273
const m87 = 1.44316060e-25
const k_B = 1.38064852e-23
const Gamma = 38107518.88804419 + 0.0*im
const Omega13 = 135886166.68478012 + 0.0*im
const Omega23 = 0.021626955125691984e9 + 0.0*im
const gamma21tilde = 321033.05667335045+42939288389.26529*im
const gamma31tilde = 46564466.011063695+61788844310.80405*im
const gamma32tilde = 46564466.011063695+18849555921.538757*im
const waist = 1.0e-3 + 0.0*im
const r0 = 1.5*waist
const window = 3.0*waist
const x0 = ComplexF64[5/8 + 0.0*im, 3/8 + 0.0*im, 0.0 + 0.0*im,
                      0.0 + 0.0*im, 0.0 + 0.0*im, 0.0 + 0.0*im,
                      0.0 + 0.0*im, 0.0+ 0.0*im]
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

@inbounds begin
@fastmath begin
function f!(dy::Array{ComplexF64, 1}, x::Array{ComplexF64, 1},
            p::Array{ComplexF64, 1}, t::Float64)
    r_sq = (p[4]+p[2]*p[1]*t - p[13])*(p[4]+p[2]*p[1]*t - p[13]) + (p[5]+p[3]*p[1]*t - p[13])*(p[5]+p[3]*p[1]*t - p[13])
    Om23 = p[8] * exp(-r_sq/(2.0*p[12]*p[12]))
    Om13 = p[7] * exp(-r_sq/(2.0*p[12]*p[12]))
    dy[1] = (-p[6]/2.0)*x[1]-(p[6]/2.0)*x[2]+(im*conj(Om13)/2.0)*x[5]-(im*Om13/2)*x[6]+p[6]/2.0
    dy[2] = (-p[6]/2.0)*x[1]-(p[6]/2.0)*x[2]+(im*conj(Om23)/2.0)*x[7]-(im*Om23/2)*x[8]+p[6]/2.0
    dy[3] = -p[9]*x[3]+(im*conj(Om23)/2.0)*x[5]-(im*Om13/2.0)*x[8]
    dy[4] = -conj(p[9])*x[4] - (im*Om23/2.0)*x[6] + (im*conj(Om13)/2.0)*x[7]
    dy[5] = im*Om13*x[1] + (im*Om13/2.0)*x[2] + (im*Om23/2.0)*x[3] - p[10]*x[5]-im*Om13/2.0
    dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2.0)*x[2]-(im*conj(Om23)/2.0)*x[4]-conj(p[10])*x[6]+im*conj(Om13)/2.0
    dy[7] = (im*Om23/2.0)*x[1]+im*Om23*x[2]+(im*Om13/2.0)*x[4]-p[11]*x[7] - im*Om23/2.0
    dy[8] = (-im*conj(Om23)/2.0)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2.0)*x[3]-conj(p[11])*x[8]+im*conj(Om23)/2.0
end
end
end


@inbounds begin
@fastmath begin
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
end
end

grid = zeros(ComplexF64, (N_grid, N_grid))
counter_grid = zeros(Int32, (N_grid, N_grid))
counter_grid_total = zeros(Int32, (N_grid, N_grid))
grid_weighted = zeros(ComplexF64, (N_grid, N_grid))
normalized = zeros(Float64, (N_grid, N_grid))
Vs = collect(LinRange{Float64}(v0, v1, N_v))
pv = sqrt(2.0/pi)*((m87/(k_B*T))^(3.0/2.0)).*Vs.^2.0 .*exp.(-m87*Vs.^2.0/(2.0*k_B*T))
v_perps = zeros(Float64, N_real)
global coords = Array{Tuple{Int32, Int32, Int32, Int32}}(undef, N_real)
for (index_v, v) in ProgressBar(enumerate(Vs))
    paths = [[] for i=1:N_real]
    tpaths = [[] for i=1:N_real]
    # Choose points in advance to save only the relevant points during solving
    Threads.@threads for i = 1:N_real
        iinit, jinit, ifinal, jfinal = choose_points()
        coords[i] = (iinit, jinit, ifinal, jfinal)
        paths[i] = bresenham(jinit, iinit, jfinal, ifinal)
        v_perps[i] = sqrt(v^2.0 - draw_vz(v)^2.0)
        tpaths[i] = Float64[hypot(coord[2]-iinit, coord[1]-jinit)*abs(window)/(v_perps[i]*N_grid) for coord in paths[i]]
    end
    function prob_func(prob, i, repeat)
        iinit, jinit, ifinal, jfinal = coords[i]
        v_perp = v_perps[i]
        vz = sqrt(v^2.0 - v_perp^2.0)
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
        else
            u0 = u1 = 0
        end
        new_p = ComplexF64[v_perp + 0.0*im, u0 + 0.0*im, u1 + 0.0*im,
                 xinit + 0.0*im, yinit + 0.0*im,
                 Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde - im*k*vz,
                 gamma32tilde - im*k*vz, waist, r0]
        
        tsave = tpaths[i]
        new_tspan = (0.0, maximum(tsave))
        remake(prob, p=new_p, tspan=new_tspan, saveat=tsave)
    end

    # instantiate a problem
    p = ComplexF64[1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im,
         Gamma, Omega13,
         Omega23, gamma21tilde,
         gamma31tilde - im*k*0.0,
         gamma32tilde - im*k*0.0, waist, r0]
    tspan = (0.0, 1.0)
    tsave = collect(LinRange{Float64}(tspan[1], tspan[2], 2))
    prob = ODEProblem{true}(f!, x0, tspan, p, jac=f_jac!, saveat=tsave)
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    # run once on small system to try and speed up compile time
    if index_v==1
        sol = solve(ensembleprob, TRBDF2(autodiff=false), EnsembleThreads(),
                          trajectories=2, save_idxs=7)

    end
    sol = solve(ensembleprob, TRBDF2(autodiff=false), EnsembleThreads(),
                      trajectories=N_real, save_idxs=7)

    global grid .= zeros(ComplexF64, (N_grid, N_grid))
    global counter_grid .= zeros(Int32, (N_grid, N_grid))
    @inbounds begin
    Threads.@threads for i = 1:N_real
        for (j, coord) in enumerate(paths[i])
            grid[coord[2], coord[1]] += sol[i].u[j]
            counter_grid[coord[2], coord[1]] += 1
        end
    end
    end
    grid_weighted .+= (grid./counter_grid) * pv[index_v]
    counter_grid_total .+= counter_grid
    
end

pol = real.(grid_weighted)
fig, ax = subplots(1, 2)
ax[1].imshow(pol)
ax[2].imshow(counter_grid_total)
ax[1].set_title("Polarization")
ax[2].set_title("Counts")
show()
