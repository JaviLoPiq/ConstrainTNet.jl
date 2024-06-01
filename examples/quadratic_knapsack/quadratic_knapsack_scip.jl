using JuMP, SCIP, Random

function run_job_scip(seed)
    N = 100

    Random.seed!(seed)
    A = rand(0:5, 1, N)
    lb = [0]
    ub = [Int(floor(N/4))]

    Random.seed!(seed)
    Q = rand(-5:5,N,N)  

    # Create a model with SCIP as the solver
    model = Model(SCIP.Optimizer)

    # Define binary variables
    @variable(model, x[1:N], Bin)

    # Define the quadratic objective function
    @objective(model, Min, x' * Q * x)

    # Add linear constraints
    for i in 1:size(A, 1)
        @constraint(model, lb[i] <= sum(A[i, j] * x[j] for j in 1:N) <= ub[i])
    end

    set_optimizer_attribute(model, "limits/time", 500)  

    # Solve the problem
    optimize!(model)
    #=
    # Save the detailed history to a file
    open("optimization_history_scip_seed_$(seed).txt", "w") do f
        println(f, "Time(s)\tObjective Value")
        for (t, obj) in history
            println(f, "$t\t$obj")
        end
    end

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.TIME_LIMIT
        println("Final solution found:")
        println("Objective value: ", objective_value(model))
        # You can add more details here
    else
        println("Problem not solved to optimality. Status: ", termination_status(model))
    end
    =#
    # Check the solution status and print the results
    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.TIME_LIMIT
        println("Optimal solution found:")
        for i in 1:N
            println("x[$i] = ", value(x[i]))
        end
        println("Objective value: ", objective_value(model))
    else
        println("Problem not solved to optimality. Status: ", termination_status(model))
    end
end

seeds = [1, 2, 3, 4, 5]

for i in eachindex(seeds)
    println("seed: ", i)
    run_job_scip(seeds[i])
end