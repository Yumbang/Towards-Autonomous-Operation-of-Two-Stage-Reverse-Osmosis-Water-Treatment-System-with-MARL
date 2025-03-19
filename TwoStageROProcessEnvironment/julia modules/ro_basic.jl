module RO_Element_Simple

include("./RO_Utils_Module.jl")

using Printf, Statistics, DataFrames, Plots
using .RO_Utils_Module: osmo_press

mem_area = 37 * 7
length = 1.016 * 7
W = mem_area / length
H = 8.64e-4
n_segments = 700
dx = length / n_segments
# Rm0 = 5.811e13    # Calculated based on A value of the ESPA membrane (6.1753)
r = 0.995
# K = 15.9999389648438
# K = 10.0
# k_fp = 1.0e9*24*60*60

export ro_vessel_simple

function ro_vessel_simple(;state_vars::Dict, operational_vars::Dict, A_setpoint, K_setpoint, k_fp_setpoint, dt)
    # Process input parameters.
    timestep = state_vars["timestep"]
    dt = dt

    C_feed0     = operational_vars["C"] / 1e3   # kg/m3. Original dictionary value is in ppm.
    Q_feed0     = operational_vars["Q"]
    P_feed0     = operational_vars["P"] * 1e5
    U_feed0     = Q_feed0 / W / H / 3600
    T_feed0     = operational_vars["T"]

    # Exposed K and k_fp for parameter optimization.
    K = K_setpoint
    k_fp = k_fp_setpoint * 1e9*24*60*60


    A       = A_setpoint / 3600 / 1e3 / 1e5
    a_T     = 4140
    TCF_A   = exp(a_T*(1/(T_feed0 + 273.15) - 1/293.15))
    μ       = 2.414e-5 * 10^(247.8 / (T_feed0 + 273.15 - 140))
    Rm0     = 1/A

    # Define Arrays to store the updated variables

    C           = zeros(n_segments)
    v_w         = zeros(n_segments)
    U           = zeros(n_segments)
    P           = zeros(n_segments)
    R_m         = zeros(n_segments)
    R_m_updated = zeros(n_segments)
    Osmo_P      = zeros(n_segments)
    converged   = Bool[]
    
    if Int(timestep) == 1
        R_m .= Rm0
    else
        R_m = state_vars["R_m"]
    end

    # Initialize the arrays' 1st segments values
    begin
        local err               = 1.0
        local idx_iter          = 0
        local c_cal             = C_feed0
        local v_w_guess         = 0.0
        local osmo_p_guess      = 0.0
        local converged_temp    = true

        while err >= 1e-6
            c_guess = c_cal
            osmo_p_guess = osmo_press(c_guess*1000, T_feed0)
            v_w_guess = (P_feed0 - osmo_p_guess) * TCF_A / R_m[1]
            c_cal = (C_feed0 * U_feed0 * H - (1-r) * (c_guess * v_w_guess * dx)) / (U_feed0 * H)
            err = abs((c_cal - c_guess) / c_cal)
            idx_iter += 1
            if idx_iter > 1e2
                println("The number of iteration exceeded 100.")
                converged_temp = false
                break
            end
        end

        v_w[1]          = v_w_guess
        C[1]            = c_cal
        U[1]            = U_feed0 - (v_w[1]*dx)/H
        P[1]            = P_feed0 - ((12 * K * μ * U[1] * dx) / H^2)  # * U[1]
        Osmo_P[1]       = osmo_p_guess
        R_m_updated[1]  = R_m[1] + k_fp * v_w[1] * dt / 60 / 24
        push!(converged, converged_temp)
    end

    for segment_index in 2:n_segments
        local err               = 1.0
        local idx_iter          = 0
        local c_cal             = C[segment_index-1]
        local v_w_guess         = 0.0
        local osmo_p_guess      = 0.0
        local converged_temp    = true

        while err >= 1e-6
            c_guess = c_cal
            osmo_p_guess = osmo_press(c_guess*1000, T_feed0)
            v_w_guess = (P[segment_index-1] - osmo_p_guess) * TCF_A / R_m[segment_index-1]
            c_cal = (C_feed0 * U_feed0 * H - (1-r) * (c_guess * v_w_guess * dx)) / (U[segment_index-1] * H)
            err = abs((c_cal - c_guess) / c_cal)
            idx_iter += 1
            if idx_iter > 1e2
                println("The number of iteration exceeded 100.")
                converged_temp = false
                break
            end
        end

        v_w[segment_index]          = v_w_guess
        C[segment_index]            = c_cal
        U[segment_index]            = U[segment_index-1] - (v_w[segment_index]*dx)/H
        P[segment_index]            = P[segment_index-1] - ((12 * K * μ * U[segment_index] * dx) / H^2)  # * U[segment_index]
        Osmo_P[segment_index]       = osmo_p_guess
        R_m_updated[segment_index]  = R_m[segment_index] + k_fp * v_w[segment_index] * dt / 60 / 24
        push!(converged, converged_temp)
    end

    product = mean(v_w * W * length * 3600)
    recovery = product / operational_vars["Q"]
    # @printf "Feed : %.2f \nProduct : %.2f\nRecovery : %.2f\n" operational_vars["Q"]*84 product*84 recovery

    permeate_vars = Dict(
        "Q" => product,                 # [m3/hour]
        "C" => mean(C) * (1-r) * 1e3,   # [ppm]
        "T" => T_feed0,                 # [°C]
        "P" => 1e-10,                   # [bar]
    )

    brine_vars = Dict(
        "Q" => operational_vars["Q"] - permeate_vars["Q"],
        "C" => C[end] * 1e3,
        "T" => T_feed0,
        "P" => P[end] / 1e5,
    )

    state_vars_updated = Dict(
        "R_m" => R_m_updated,
        "v_total" => v_w,
        "u_total" => U,
        "cp_total" => C .* (1-r) * 1e3,
        "p_total" => P,
        "osmo_p_total" => Osmo_P,
        "converged" => converged,
        "timestep" => timestep
    )

    return [permeate_vars, brine_vars, state_vars_updated]

end

function ro_vessel_simple(;state_vars::Dict, operational_vars::Dict, parameter_setpoints::Dict, dt)
    # Process input parameters.
    timestep = state_vars["timestep"]
    dt = dt

    C_feed0     = operational_vars["C"] / 1e3   # kg/m3. Original dictionary value is in ppm.
    Q_feed0     = operational_vars["Q"]
    P_feed0     = operational_vars["P"] * 1e5
    U_feed0     = Q_feed0 / W / H / 3600
    T_feed0     = operational_vars["T"]

    # Set parameters with the received dictionary.
    # K_setpoint = parameter_setpoints["K"]
    k_fp_setpoint = parameter_setpoints["k_fp"]
    A_setpoint = parameter_setpoints["A"]

    K       = 16.0
    k_fp    = k_fp_setpoint * 1e9*24*60*60
    A       = A_setpoint / 3600 / 1e3 / 1e5
    a_T     = 4140
    TCF_A   = exp(a_T*(1/(T_feed0 + 273.15) - 1/293.15))
    μ       = 2.414e-5 * 10^(247.8 / (T_feed0 + 273.15 - 140))  # 1e-3 scale
    Rm0     = 1/A

    # Define Arrays to store the updated variables

    C           = zeros(n_segments)
    v_w         = zeros(n_segments)
    U           = zeros(n_segments)
    P           = zeros(n_segments)
    R_m         = zeros(n_segments)
    R_m_updated = zeros(n_segments)
    Osmo_P      = zeros(n_segments)
    converged   = Bool[]
    
    if Int(timestep) == 1
        R_m .= Rm0
    else
        R_m = state_vars["R_m"]
    end

    # Initialize the arrays' 1st segments values
    begin
        local err               = 1.0
        local idx_iter          = 0
        local c_cal             = C_feed0
        local v_w_guess         = 0.0
        local osmo_p_guess      = 0.0
        local converged_temp    = true

        while err >= 1e-6
            c_guess = c_cal
            osmo_p_guess = osmo_press(c_guess*1000, T_feed0)
            v_w_guess = (P_feed0 - osmo_p_guess) / R_m[1] / TCF_A
            c_cal = (C_feed0 * U_feed0 * H - (1-r) * (c_guess * v_w_guess * dx)) / (U_feed0 * H)
            err = abs((c_cal - c_guess) / c_cal)
            idx_iter += 1
            if idx_iter > 1e2
                println("The number of iteration exceeded 100.")
                converged_temp = false
                break
            end
        end

        v_w[1]          = v_w_guess
        C[1]            = c_cal
        U[1]            = U_feed0 - (v_w[1]*dx)/H
        P[1]            = P_feed0 - ((12 * K * μ * U[1] * dx) / H^2)  # * U[1]
        Osmo_P[1]       = osmo_p_guess
        R_m_updated[1]  = R_m[1] + k_fp * v_w[1] * dt / 60 / 24
        push!(converged, converged_temp)
    end

    # C[1] = C_feed0
    # U[1] = U_feed0
    # P[1] = P_feed0


    # v_w[1] = (P[1] - osmo_press(C[1], T_feed0)) / R_m[1] / TCF_A 
    # R_m_updated[1] = R_m[1] + k_fp * v_w[1] * dt / 60 / 24

    for segment_index in 2:n_segments
        local err               = 1.0
        local idx_iter          = 0
        local c_cal             = C[segment_index-1]
        local v_w_guess         = 0.0
        local osmo_p_guess      = 0.0
        local converged_temp    = true

        while err >= 1e-6
            c_guess = c_cal
            osmo_p_guess = osmo_press(c_guess*1000, T_feed0)
            v_w_guess = (P[segment_index-1] - osmo_p_guess) / R_m[segment_index-1] / TCF_A
            c_cal = (C_feed0 * U_feed0 * H - (1-r) * (c_guess * v_w_guess * dx)) / (U[segment_index-1] * H)
            err = abs((c_cal - c_guess) / c_cal)
            idx_iter += 1
            if idx_iter > 1e2
                println("The number of iteration exceeded 100.")
                converged_temp = false
                break
            end
        end

        v_w[segment_index]          = v_w_guess
        C[segment_index]            = c_cal
        U[segment_index]            = U[segment_index-1] - (v_w[segment_index]*dx)/H
        P[segment_index]            = P[segment_index-1] - ((12 * K * μ * U[segment_index] * dx) / H^2)  # * U[segment_index]
        Osmo_P[segment_index]       = osmo_p_guess
        R_m_updated[segment_index]  = R_m[segment_index] + k_fp * v_w[segment_index] * dt / 60 / 24
        push!(converged, converged_temp)
    end

    product = mean(v_w * W * length * 3600)
    recovery = product / operational_vars["Q"]
    # @printf "Feed : %.2f \nProduct : %.2f\nRecovery : %.2f\n" operational_vars["Q"]*84 product*84 recovery

    permeate_vars = Dict(
        "Q" => product,                 # [m3/hour]
        "C" => mean(C) * (1-r) * 1e3,   # [ppm]
        "T" => T_feed0,                 # [°C]
        "P" => 1e-10,                   # [bar]
    )

    brine_vars = Dict(
        "Q" => operational_vars["Q"] - permeate_vars["Q"],
        "C" => C[end] * 1e3,
        "T" => T_feed0,
        "P" => P[end] / 1e5,
    )

    state_vars_updated = Dict(
        "R_m" => R_m_updated,
        "v_total" => v_w,
        "u_total" => U,
        "cp_total" => C .* (1-r) * 1e3,
        "p_total" => P,
        "osmo_p_total" => Osmo_P,
        "converged" => converged,
        "timestep" => timestep
    )

    return [permeate_vars, brine_vars, state_vars_updated]

end

end