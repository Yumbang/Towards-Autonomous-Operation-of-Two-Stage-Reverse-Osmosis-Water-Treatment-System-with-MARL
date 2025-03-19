module RecoveryControlledRO

include("./ro_basic.jl")
# using .RO_Element_Simple
using DataFrames
using Printf
using DiscretePIDs
using Random
using PythonCall
using Statistics
const ro_1st_pvs = 84.0
const ro_2nd_pvs = 48.0
k_fp_1st_setpoint   = 0.674150595374451
k_fp_2nd_setpoint   = 1.9727956135868776
K_setpoint          = 16.0
A_setpoint          = 5.295039332042272
RO_1st_setpoints = Dict(
    "k_fp"  => k_fp_1st_setpoint,
    "A"     => A_setpoint
)
RO_2nd_setpoints = Dict(
    "k_fp"  => k_fp_2nd_setpoint,
    "A"     => A_setpoint
)
dt = 6.0

export recovery_controlled_2stage_ro_simple

function calculate_SEC(feed_Q_sum, applied_pressure, product_Q_sum, pump_efficiency)
    power_required = applied_pressure * 1e5 * feed_Q_sum / 60 / 60 / pump_efficiency / 1e3
    sec = power_required / product_Q_sum
    return sec
end

function ramp_setpoint(setpoint_old, setpoint_new, ramp_length, ramp_index)
    if ramp_index > ramp_length
        return setpoint_new
    else
        return (setpoint_old * (ramp_length - ramp_index) + setpoint_new * ramp_index) / ramp_length
    end
end

function recovery_controlled_2stage_ro_simple(feed_scenario::Matrix, flowrate::Float64,
    recovery_1st_setpoint::Float64, recovery_2nd_setpoint::Float64, flowrate_old::Float64, recovery_1st_setpoint_old::Float64, recovery_2nd_setpoint_old::Float64,
    st_var_1st::Dict, st_var_2nd::Dict, HPP_initial, IBP_initial, 
    pid_controller_1st::DiscretePID, pid_controller_2nd::DiscretePID)
    """
    feed_scenario: Matrix with "T", "C" and "P_in" as columns and data points as rows.
    flowrate: Flowrate value to use.
    recovery: Desired recovery rate.
    state_var: Dictionary with ro element state variables
    """
    step = size(feed_scenario)[1]

    P_1st_initial = min(13.3 - feed_scenario[1, 1] * 0.35, 2.5)

    permeate_1st_log = Any[]
    permeate_2nd_log = Any[]
    brine_1st_log = Any[]
    brine_2nd_log = Any[]
    recovery_1st_log = Any[]
    recovery_2nd_log = Any[]

    op_var_1st_log = Any[]
    op_var_2nd_log = Any[]
    state_1st_log = Any[]
    state_2nd_log = Any[]

    # Store calculated SEC, in [kWh/m3]
    SEC_1st_log = Any[]
    SEC_2nd_log = Any[]
    SEC_total_log = Any[]

    state_var_1st = st_var_1st
    state_var_2nd = st_var_2nd

    converged = true

    # stabilization_period = 1:10

    for i in 1:step
        if (i == 1) # | (st_var_1st["timestep"] in stabilization_period)
            if isnothing(HPP_initial)
                HPP_1st = P_1st_initial
            else
                HPP_1st = HPP_initial
            end
        else
            recovery_1st_ramped_setpoint = ramp_setpoint(recovery_1st_setpoint_old, recovery_1st_setpoint, 4, i)
            HPP_1st = calculate_control!(pid_controller_1st, recovery_1st_ramped_setpoint, recovery_1st_log[end])
        end

        flowrate_ramped = ramp_setpoint(flowrate_old, flowrate, 4, i)

        op_var_1st = Dict(
            "T" => feed_scenario[i, 1],
            "C" => feed_scenario[i, 2],
            "Q" => flowrate_ramped / ro_1st_pvs,
            "P" => max(feed_scenario[i, 3] + HPP_1st, 2.5),
            # "C_CF" => feed_scenario[i, 4]
        )
        push!(op_var_1st_log, op_var_1st)

        permeate_1st, brine_1st, state_var_1st_updated = RO_Element_Simple.ro_vessel_simple(;operational_vars=op_var_1st, state_vars=state_var_1st, parameter_setpoints=RO_1st_setpoints, dt=dt)

        recovery_1st = permeate_1st["Q"] / op_var_1st["Q"]
        permeate_1st["Q"] *= ro_1st_pvs
        brine_1st["Q"] *= ro_1st_pvs
        sec_1st = calculate_SEC(op_var_1st["Q"]*ro_1st_pvs, op_var_1st["P"]-feed_scenario[i,3], permeate_1st["Q"], 0.8)
        state_var_1st = state_var_1st_updated
        
        push!(recovery_1st_log, recovery_1st)
        push!(permeate_1st_log, permeate_1st)
        push!(brine_1st_log, brine_1st)
        push!(SEC_1st_log, sec_1st)
        push!(state_1st_log, state_var_1st)

        op_var_2nd = copy(brine_1st)

        op_var_2nd["Q"] /= ro_2nd_pvs

        if (i == 1) # | (st_var_2nd["timestep"] in stabilization_period)
            if isnothing(IBP_initial)
                IBP_2nd = 0.0
            else
                IBP_2nd = IBP_initial
            end
        else
            recovery_2nd_ramped_setpoint = ramp_setpoint(recovery_2nd_setpoint_old, recovery_2nd_setpoint, 4, i)
            IBP_2nd = calculate_control!(pid_controller_2nd, recovery_2nd_ramped_setpoint, recovery_2nd_log[end])
        end
        
        op_var_2nd["P"] += IBP_2nd
        push!(op_var_2nd_log, op_var_2nd)

        permeate_2nd, brine_2nd, state_var_2nd_updated = RO_Element_Simple.ro_vessel_simple(;operational_vars=op_var_2nd, state_vars=state_var_2nd, parameter_setpoints=RO_2nd_setpoints, dt=dt)

        recovery_2nd = permeate_2nd["Q"] / op_var_2nd["Q"]
        permeate_2nd["Q"] *= ro_2nd_pvs
        brine_2nd["Q"] *= ro_2nd_pvs
        sec_2nd = calculate_SEC(op_var_2nd["Q"]*ro_2nd_pvs, IBP_2nd, permeate_2nd["Q"], 0.8)
        state_var_2nd = state_var_2nd_updated
        
        push!(recovery_2nd_log, recovery_2nd)
        push!(permeate_2nd_log, permeate_2nd)
        push!(brine_2nd_log, brine_2nd)
        push!(SEC_2nd_log, sec_2nd)
        push!(state_2nd_log, state_var_2nd_updated)

        sec_total = (sec_1st * permeate_1st["Q"] + sec_2nd * permeate_2nd["Q"]) / (permeate_1st["Q"] + permeate_2nd["Q"])
        push!(SEC_total_log, sec_total)

        state_var_1st["timestep"] += 1
        state_var_2nd["timestep"] += 1

        # Divergence or model malfunction detection.

        if !(all(state_var_1st["converged"]) & all(state_var_2nd["converged"]))
            @printf("Process diverged at timestep %d\n", state_var_1st["timestep"])
            converged = false
            break
        end

        if op_var_1st["P"] > 39.0
            @printf("Model malfunction detected at timestep %d\n", state_var_1st["timestep"])
            converged = false
            break
        end
    end

    step_proceed = length(op_var_1st_log)
    blackbox_1st = state_1st_log[end - min(step_proceed-1, 10) : end]
    blackbox_2nd = state_2nd_log[end - min(step_proceed-1, 10) : end]

    return state_var_1st, state_var_2nd, permeate_1st_log, permeate_2nd_log, brine_1st_log, brine_2nd_log, recovery_1st_log, recovery_2nd_log, op_var_1st_log, op_var_2nd_log, blackbox_1st, blackbox_2nd, SEC_1st_log, SEC_2nd_log, SEC_total_log, converged
end

end

# begin
#     using DiscretePIDs
#     using .RecoveryControlledRO
#     using Printf

#     feed_scenario = cat(
#         ones(1440) * 15.0, ones(1440) * 600.0, ones(1440) * 1e-3, ones(1440) * 1e-3; dims = 2
#     )

#     state_1st_snapshot = []
#     state_2nd_snapshot = []
#     op_var_1st_total_log = []
#     op_var_2nd_total_log = []
#     perm_1st_total_log = []
#     perm_2nd_total_log = []
#     flowrate = 1000.0
#     recovery_1st_setpoint = 0.75
#     recovery_2nd_setpoint = 0.50
#     st_var_1st = Dict(
#         "timestep" => 1.0
#     )
#     st_var_2nd = Dict(
#         "timestep" => 1.0
#     )
#     pid_controller_1st = DiscretePID(; K=0.5, Ti=1, Td=1, Ts=1.0, umin=0.0, umax=20.0)
#     pid_controller_2nd = DiscretePID(; K=0.5, Ti=1, Td=1, Ts=1.0, umin=0.0, umax=20.0)

#     for days in 1:10
#         state_var_1st, state_var_2nd, permeate_1st_log, permeate_2nd_log, brine_1st_log, brine_2nd_log, recovery_1st_log, recovery_2nd_log, op_var_1st_log, op_var_2nd_log, blackbox_1st, blackbox_2nd, converged = recovery_controlled_2stage_ro(feed_scenario, flowrate, recovery_1st_setpoint, recovery_2nd_setpoint, st_var_1st, st_var_2nd, pid_controller_1st, pid_controller_2nd)
#         st_var_1st = state_var_1st
#         st_var_2nd = state_var_2nd
#         push!(state_1st_snapshot, st_var_1st)
#         push!(state_2nd_snapshot, st_var_2nd)
#         push!(op_var_1st_total_log, op_var_1st_log)
#         push!(op_var_2nd_total_log, op_var_2nd_log)
#         push!(perm_1st_total_log, permeate_1st_log)
#         push!(perm_2nd_total_log, permeate_2nd_log)
#         # op_1st_plot = plot([op_1st["P"] for op_1st in op_var_1st_log], title=(@sprintf("Pressure of 1st RO in day %d",days)))
#         # op_2nd_plot = plot([op_2nd["P"] for op_2nd in op_var_2nd_log], title=(@sprintf("Pressure of 2nd RO in day %d",days)))
#         # savefig(op_1st_plot, @sprintf("Day_%d_1st_RO.png", days))
#         # savefig(op_2nd_plot, @sprintf("Day_%d_2nd_RO.png", days))
#     end

#     using Plots
#     plot(dropdims(state_2nd_snapshot[1]["H_local"]; dims=1))
#     plot(dropdims(state_2nd_snapshot[10]["H_local"]; dims=1))

#     plot(state_1st_snapshot[1]["v_total"])
#     plot(state_2nd_snapshot[1]["v_total"])

#     state_2nd_snapshot[1]["v_total"]
#     state_2nd_snapshot[13]["v_total"]
#     plot(state_2nd_snapshot[1]["v_total"] .- state_2nd_snapshot[10]["v_total"])

#     plot([perm["Q"] for perm in perm_1st_total_log[end]])
#     plot([perm["Q"] for perm in perm_2nd_total_log[end]])
# end