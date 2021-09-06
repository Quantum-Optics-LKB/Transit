using DifferentialEquations
using Plots

# global variables to be set from python side through Julia "Main" namespace
N_real = 5000
N_grid = 256
T = 150+273
m87 = 1.44316060e-25
k_B = 1.38064852e-23
window = 2.5e-3
Gamma = 1.0 + 0*im
Omega13 = 1.0 + 0*im
Omega23 = 1.0 + 0*im
gamma21tilde = 1.0 + 0*im
gamma31tilde = 1.0 + 0*im
gamma32tilde = 1.0 + 0*im
waist = 1.0e-3 + 0*im
r0 = 1.0 + 0*im
x0 = [5/8 + 0*im, 3/8+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im, 0+ 0*im]
k = 2*pi/780.241e-9
v = 40.0

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
        append!(points, coord)
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
    for i in 0:N_grid-1
        edges[i+1] = (0, i)
        edges[i+1+N_grid] = (N_grid-1, i)
        edges[i+1+2*N_grid] = (i, 0)
        edges[i+1+3*N_grid] = (i, N_grid-1)
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

function f!(dy, x, p, t)
    v, u0, u1, xinit, yinit = p[1], p[2], p[3], p[4], p[5]
    Gamma, Omega13, Omega23 = p[6], p[7], p[8]
    gamma21tilde, gamma31tilde, gamma32tilde = p[9], p[10], p[11]
    waist, r0 = p[12], p[13]
    r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) + (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
    Om23 = Omega23 * exp(-r_sq/(2*waist*waist))
    Om13 = Omega13 * exp(-r_sq/(2*waist*waist))
    dy[1] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om13)/2)*x[5]-(im*Om13/2)*x[6]+Gamma/2
    dy[2] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om23)/2)*x[7]-(im*Om23/2)*x[8]+Gamma/2
    dy[3] = -gamma21tilde*x[3]+(im*conj(Om23)/2)*x[5]-(im*Om13/2)*x[8]
    dy[4] = -conj(gamma21tilde)*x[4] - (im*Om23/2)*x[6] + (im*conj(Om13)/2)*x[7]
    dy[5] = im*Om13*x[1] + (im*Om13/2)*x[2] + (im*Om23/2)*x[3] - gamma31tilde*x[5]-im*Om13/2
    dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2)*x[2]-(im*conj(Om23)/2)*x[4]-conj(gamma31tilde)*x[6]+im*conj(Om13)/2
    dy[7] = (im*Om23/2)*x[1]+im*Om23*x[2]+(im*Om13/2)*x[4]-gamma32tilde*x[7] - im*Om23/2
    dy[8] = (-im*conj(Om23)/2)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2)*x[3]-conj(gamma32tilde)*x[8]+im*conj(Om23)/2
end

paths = [[] for i=1:N_real]
xs = [[] for i=1:N_real]
ts = [[] for i=1:N_real]

function prob_func(prob, i, repeat)
    iinit, jinit, ifinal, jfinal = choose_points()
    path = bresenham(jinit, iinit, jfinal, ifinal)
    paths[i] = path
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
    new_p = [v_perp + 0*im, u0 + 0*im, u1 + 0*im, xinit + 0*im, yinit + 0*im,
         Gamma, Omega13,
         Omega23, gamma21tilde,
         gamma31tilde - im*k*vz,
         gamma32tilde - im*k*vz, waist, r0]

    @. prob.p .= new_p
    @. prob.tspan .= (0.0, new_tfinal)
    prob
end

iinit, jinit, ifinal, jfinal = choose_points()
path = bresenham(jinit, iinit, jfinal, ifinal)
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
    tfinal = hypot((xfinal-xinit), (yfinal-yinit))/v_perp
else
    u0 = u1 = 0
    tfinal = hypot((xfinal-xinit), (yfinal-yinit))/abs(vz)
end
p = [v_perp + 0*im, u0 + 0*im, u1 + 0*im, xinit + 0*im, yinit + 0*im,
     Gamma, Omega13,
     Omega23, gamma21tilde,
     gamma31tilde - im*k*vz,
     gamma32tilde - im*k*vz, waist, r0]
tspan = (0.0, tfinal)
tsave = collect(LinRange(0.0, tfinal, 1000))
prob = ODEProblem(f!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)
sol = solve(ensembleprob, BS3(), EnsembleThreads(), saveat=tsave, trajectories=N_real)
for i in 1:N_real
    xs[i] = sol[i].u
    ts[i] = sol[i].t
end
