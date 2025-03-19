# Set the path below as your desired path including ro_basic.jl file. 
cd("/home/ybang4/research/ROMARL/TwoStageROProcessEnvironment/julia modules")
include("./ro_basic.jl")
using .RO_Element_Simple
using Pkg
Pkg.add(["DataFrames", "Printf", "CSV", "Plots", "BlackBoxOptim"])
using DataFrames
using Printf
using Random
using Statistics
using CSV
using Plots
using StatsBase
using BlackBoxOptim
using JLD2
using XLSX

println(Threads.nthreads())

const ro_1st_pvs = 84.0
const ro_2nd_pvs = 48.0
const H = 8.64e-4

function calculate_SEC(feed_Q_sum, applied_pressure, product_Q_sum, pump_efficiency)
    power_required = applied_pressure * 1e5 * feed_Q_sum / 60 / 60 / pump_efficiency / 1e3
    sec = power_required / product_Q_sum
    return sec
end

function MAPE(;y_pred, y_obs)
    if size(y_pred) != size(y_obs)
        error("The size of the input variables are not matching (y_pred: $(size(y_pred)) / y_obs: $(size(y_obs)))")
    else
        mape = mean((y_pred .- y_obs) ./ y_obs) * 100
        return mape
    end
end

function r2_score(y_true, y_pred)
    # Mean of the true values
    mean_y_true = mean(y_true)
    
    # Total Sum of Squares (TSS)
    tss = sum((y_true .- mean_y_true) .^ 2)
    
    # Residual Sum of Squares (RSS)
    rss = sum((y_true .- y_pred) .^ 2)
    
    # R² score
    r2 = 1 - (rss / tss)
    
    return r2
end

global FC = 0

function calculate_RMSE(setpoint)
    save_fig = false
    parameter_setpoints_1st = Dict(
        "k_fp" => setpoint[1],
        "A" => setpoint[3]  #setpoint[4]
    )
    parameter_setpoints_2nd = Dict(
        "k_fp" => setpoint[2],
        "A" => setpoint[3]  #setpoint[4]
    )
    Daesan_path = "./Data/Daesan_$FC.xlsx"
    save_root_path = "./Result/simple_RO/Optimizing_k_fp_Daesan$FC"

    # daesan = CSV.read(Daesan_path, DataFrame)
    daesan = DataFrame(XLSX.readtable(Daesan_path, "Sheet2"))
    dropmissing!(daesan)
    step = nrow(daesan)

    feed_scenario = daesan[:, [3,4,5,6]]
    feed_scenario.유입Foulant .= 1e-2

    state_var_1st = Dict(
        "timestep" => 1.0
    )
    state_var_2nd = Dict(
        "timestep" => 1.0
    )

    daesan_dt = 60.0
    # dt = 3.0
    dt = daesan_dt
    step_multiplier = Int(daesan_dt/dt)

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

    converged = true

    for i in 1:step

        op_var_1st = Dict(
            "T"     => feed_scenario[i, 1],
            "C"     => feed_scenario[i, 2],
            "Q"     => feed_scenario[i, 3] / ro_1st_pvs,
            "P"     => feed_scenario[i, 4],
            "C_CF"  => feed_scenario[i, 5]
        )

        for micro_step in 1:step_multiplier
            push!(op_var_1st_log, op_var_1st)
            permeate_1st, brine_1st, state_var_1st_updated = RO_Element_Simple.ro_vessel_simple(;state_vars = state_var_1st, operational_vars = op_var_1st, parameter_setpoints = parameter_setpoints_1st, dt=dt)


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
            push!(op_var_2nd_log, op_var_2nd)

            permeate_2nd, brine_2nd, state_var_2nd_updated = RO_Element_Simple.ro_vessel_simple(;state_vars = state_var_2nd, operational_vars = op_var_2nd, parameter_setpoints = parameter_setpoints_2nd, dt=dt)

            recovery_2nd = permeate_2nd["Q"] / op_var_2nd["Q"]
            permeate_2nd["Q"] *= ro_2nd_pvs
            brine_2nd["Q"] *= ro_2nd_pvs
            # sec_2nd = calculate_SEC(op_var_2nd["Q"]*ro_2nd_pvs, IBP_2nd, permeate_2nd["Q"], 0.8)
            state_var_2nd = state_var_2nd_updated
            
            push!(recovery_2nd_log, recovery_2nd)
            push!(permeate_2nd_log, permeate_2nd)
            push!(brine_2nd_log, brine_2nd)
            # push!(SEC_2nd_log, sec_2nd)
            push!(state_2nd_log, state_var_2nd_updated)

            sec_total = sec_1st # * permeate_1st["Q"] + sec_2nd * permeate_2nd["Q"]) / (permeate_1st["Q"] + permeate_2nd["Q"])
            push!(SEC_total_log, sec_total)

            state_var_1st["timestep"] += 1
            state_var_2nd["timestep"] += 1

            if !(all(state_var_1st["converged"]) & all(state_var_2nd["converged"]))
                @printf("Process diverged at timestep %d\n", state_var_1st["timestep"])
                converged = false
                break
            end
        end
        if !converged
            break
        end
        # @printf "Step %d done\n" i
    end

    save_root_path = save_root_path * @sprintf "/%dmin" Int(dt)
    mkpath(save_root_path)

    permeate_1st_df = DataFrame(permeate_1st_log)
    permeate_2nd_df = DataFrame(permeate_2nd_log)

    permeate_total_Q = permeate_1st_df.Q + permeate_2nd_df.Q

    # RMSE = L2dist(permeate_total_Q, daesan.생산유량)
    # MAPE = MAPE(;y_pred = permeate_total_Q, y_obs = daesan.생산유량)
    R2 = r2_score(daesan.생산유량, permeate_total_Q)

    return -1.0 * R2
end

using BlackBoxOptim: num_func_evals

optimization_log = []

function callback_record_progress(optcontroller)
    optim_log = Dict(
        "N of function evaluation" => num_func_evals(optcontroller),
        "Best fitness (R2)" => best_fitness(optcontroller),
        "Best k_fp_1st" => best_candidate(optcontroller)[1],
        "Best k_fp_2nd" => best_candidate(optcontroller)[2],
        # "Best K" => best_candidate(optcontroller)[3],
        "Best A" => best_candidate(optcontroller)[3],
    )
    push!(optimization_log, optim_log)
    for item in optim_log
        print(item[1] * " : " * @sprintf("%.2f",item[2]) * " | ")
    end
    println("")
end


function optimize_RMSE()
    # Optimize k_fp_1st, k_fp_2nd, K and A
    # opt = bbsetup(calculate_RMSE; Method=:xnes, SearchRange = [(0.5, 5.0), (0.5, 5.0), (0.5, 25.0), (1.0, 15.0)]
    # good_guess = [0.82, 1.55, 6.72]
    opt = bbsetup(calculate_RMSE; Method=:xnes, SearchRange = [(0.5, 7.5), (0.5, 7.5), (1.0, 15.0)]
                , MaxSteps=500, NThreads = Threads.nthreads()-1, TraceInterval=1.0, TraceMode=:silent
                , CallbackFunction = callback_record_progress, CallbackInterval = 0.0)
    result = bboptimize(opt)
    return result
end

for i in [1,4,6]
    global FC = i
    result = optimize_RMSE()
    optimization_df = DataFrame(optimization_log)
    CSV.write(@sprintf("parameter_optimization_result_daesan%d_fixed_K.csv", FC), optimization_df)
    global optimization_log = []
end
# optimization_df = CSV.read("parameter_optimization_result_daesan1.csv", DataFrame)
# optimization_plot = plot(optimization_df[:, "N of function evaluation"], optimization_df[:, "Best fitness (RMSE)"])
# savefig(optimization_plot, "/home/ybang/research/ROMARL/TwoStageROProcessEnvironment/julia modules/Result/simple_RO/Optimizing_k_fp_Daesan1/60min/optimization.png")

# daesan_5 = CSV.read("Data/Daesan_5.csv", DataFrame)
# println(names(daesan_5))