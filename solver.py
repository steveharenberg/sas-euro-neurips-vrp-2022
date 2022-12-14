# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import subprocess
import sys
import os
import uuid
import platform
import numpy as np
import time

import tools
from environment import VRPEnvironment, ControllerEnvironment
from baselines.strategies import STRATEGIES

ALL_HGS_ARGS = [
    "nbGranular",
    "fractionGeneratedNearest",
    "fractionGeneratedFurthest",
    "fractionGeneratedSweep",
    "fractionGeneratedRandomly",
    "minSweepFillPercentage",
    "maxToleratedCapacityViolation",
    "maxToleratedTimeWarp",
    "initialTimeWarpPenalty",
    "penaltyBooster",
    "minimumPopulationSize",
    "generationSize",
    "nbElite",
    "nbClose",
    "targetFeasible",
    "repairProbability",
    "growNbGranularAfterNonImprovementIterations",
    "growNbGranularAfterIterations",
    "growNbGranularSize",
    "growPopulationAfterNonImprovementIterations",
    "growPopulationAfterIterations",
    "growPopulationSize",
    "intensificationProbabilityLS",
    "diversityWeight",
    "useSwapStarTW",
    "skipSwapStarDist",
    "circleSectorOverlapToleranceDegrees",
    "minCircleSectorSizeDegrees",
    "preprocessTimeWindows",
    "useDynamicParameters",
    "randomGenerator",
]

def solve_static_vrptw(instance, time_limit=3600, tmp_dir="tmp", seed=1, args=None):
    start_time = time.time()
    time_cost = [] # stores all solution costs and the time at which they were found
    # Prevent passing empty instances to the static solver, e.g. when
    # strategy decides to not dispatch any requests for the current epoch
    if instance['coords'].shape[0] <= 1:
        yield [], 0
        time_cost.append((time.time()-start_time, 0))
        return

    if instance['coords'].shape[0] <= 2:
        solution = [[1]]
        cost = tools.validate_static_solution(instance, solution)
        yield solution, cost
        time_cost.append((time.time()-start_time, cost))
        return

    os.makedirs(tmp_dir, exist_ok=True)
    instance_filename = os.path.join(tmp_dir, "problem.vrptw")
    tools.write_vrplib(instance_filename, instance, is_vrptw=True)
    
    executable = os.path.join('baselines', 'hgs_vrptw', 'genvrp')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    # No longer need to subtract two seconds from the time limit to account for writing of the instance and delay in enforcing the time limit by HGS
    hgs_max_time = max(time_limit, 1)
    argList = [ 'timeout', str(hgs_max_time),
                executable, instance_filename, str(hgs_max_time), 
                '-seed', str(seed), '-veh', '-1', '-useWallClockTime', '1'
            ]
    
    # Add all the tunable HGS args
    if args is not None:
        vargs = vars(args)
        for hgs_arg in ALL_HGS_ARGS:
            if hgs_arg in vargs and vargs[hgs_arg] is not None:
                argList += [f'-{hgs_arg}', str(vargs[hgs_arg])]
    # log(' '.join(argList))
    with subprocess.Popen(argList, stdout=subprocess.PIPE, text=True) as p:
        routes = []
        for line in p.stdout:
            line = line.strip()
            # Parse only lines which contain a route
            if line.startswith('Route'):
                label, route = line.split(": ")
                route_nr = int(label.split("#")[-1])
                assert route_nr == len(routes) + 1, "Route number should be strictly increasing"
                routes.append([int(node) for node in route.split(" ")])
            elif line.startswith('Cost'):
                # End of solution
                solution = routes
                cost = int(line.split(" ")[-1].strip())
                check_cost = tools.validate_static_solution(instance, solution)
                assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                yield solution, cost
                time_cost.append((time.time()-start_time, cost))
                # Start next solution
                routes = []
            elif line.startswith('INFO'):
                log(line)
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with imcomplete solution (is the line with Cost missing?)"
    if 'logTimeCost' in args and args.logTimeCost:
        log("time\tcost")
        for row in time_cost:
            log(f"{row[0]:0.3f}\t{row[1]}")

def run_oracle(args, env):
    # Oracle strategy which looks ahead, this is NOT a feasible strategy but gives a 'bound' on the performance
    # Bound written with quotes because the solution is not optimal so a better solution may exist
    # This oracle can also be used as supervision for training a model to select which requests to dispatch

    # First get hindsight problem (each request will have a release time)
    done = False
    observation, info = env.reset()
    epoch_tlim = info['epoch_tlim']
    while not done:
        # Dummy solution: 1 route per request
        epoch_solution = [[request_idx] for request_idx in observation['epoch_instance']['request_idx'][1:]]
        observation, reward, done, info = env.step(epoch_solution)
    hindsight_problem = env.get_hindsight_problem()

    log(f"Start computing oracle solution with {len(hindsight_problem['coords'])} requests...")
    oracle_solution = min(solve_static_vrptw(hindsight_problem, time_limit=epoch_tlim, tmp_dir=args.tmp_dir), key=lambda x: x[1])[0]
    oracle_cost = tools.validate_static_solution(hindsight_problem, oracle_solution)
    log(f"Found oracle solution with cost {oracle_cost}")

    total_reward = run_baseline(args, env, oracle_solution=oracle_solution)
    assert -total_reward == oracle_cost, "Oracle solution does not match cost according to environment"
    return total_reward


def run_baseline(args, env, oracle_solution=None):

    rng = np.random.default_rng(args.solver_seed)
    solver_seed = args.solver_seed

    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    observation, static_info = env.reset()
    
    if not static_info['is_static'] and 'solver_seed_dynamic' in args:
        rng = np.random.default_rng(args.solver_seed_dynamic)
        solver_seed = args.solver_seed_dynamic
    
    epoch_tlim = static_info['epoch_tlim']
    # Use contest qualification phase time limits
    if epoch_tlim == 0:
        if static_info['start_epoch'] == static_info['end_epoch']:
            epoch_instance = observation['epoch_instance']
            n_cust = len(epoch_instance['request_idx']) - 1
            if n_cust <= 300:
                args.epoch_tlim = 3*60
            elif n_cust <= 500:
                args.epoch_tlim = 5*60
            else:
                args.epoch_tlim = 8*60
        else:
            args.epoch_tlim = 1*60
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']

        if args.verbose:
            log(f"Epoch {static_info['start_epoch']} <= {observation['current_epoch']} <= {static_info['end_epoch']}", newline=False)
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_new_requests = num_requests_open - num_requests_postponed
            log(f" | Requests: +{num_new_requests:3d} = {num_requests_open:3d}, {epoch_instance['must_dispatch'].sum():3d}/{num_requests_open:3d} must-go...", newline=False, flush=True)

        
        if oracle_solution is not None:
            request_idx = set(epoch_instance['request_idx'])
            epoch_solution = [route for route in oracle_solution if len(request_idx.intersection(route)) == len(route)]
            cost = tools.validate_dynamic_epoch_solution(epoch_instance, epoch_solution)
        else:
            # Select the requests to dispatch using the strategy
            # TODO improved better strategy (machine learning model?) to decide which non-must requests to dispatch
            if 'thresholdSchedule' in args and args.thresholdSchedule is not None:
                threshold_schedule = [float(x) for x in args.thresholdSchedule.split(',')]
                epoch_instance_dispatch = STRATEGIES[args.strategy]({**epoch_instance, 'observation': observation, 'static_info': static_info}, rng, threshold_schedule)
            else:
                epoch_instance_dispatch = STRATEGIES[args.strategy]({**epoch_instance, 'observation': observation, 'static_info': static_info}, rng)

            # Run HGS with time limit and get last solution (= best solution found)
            # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
            # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
            solutions = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, tmp_dir=args.tmp_dir, seed=solver_seed, args=args))
            assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
            epoch_solution, cost = solutions[-1]
            
            dynamic_prune = 'pruneRoutes' in args and args.pruneRoutes
            debug_printing = False
        
            if dynamic_prune and not static_info['is_static']:
                must_go_mask = epoch_instance_dispatch['must_dispatch']
                matrix = epoch_instance_dispatch['duration_matrix']
                n_must_go = sum(must_go_mask)
                optional = ~must_go_mask
                mask = np.ones_like(epoch_instance_dispatch['must_dispatch']).astype(np.bool8)
                optional[0] = False # depot
                n_optional = sum(optional)
                epoch_solution_pruned = epoch_solution.copy()
                for route in epoch_solution:
                    if optional[route].all():
                        mask[route] = False
                        epoch_solution_pruned.remove(route)
                        if debug_printing:
                            log(f'Pruning route {route}')
                epoch_instance_dispatch_pruned = tools._filter_instance(epoch_instance_dispatch, mask)
                epoch_solution = epoch_solution_pruned
                cost = tools.validate_static_solution(epoch_instance_dispatch, epoch_solution, allow_skipped_customers=True)

            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]
        
        if args.verbose:
            num_requests_dispatched = sum([len(route) for route in epoch_solution])
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_requests_postponed = num_requests_open - num_requests_dispatched
            log(f" {num_requests_dispatched:3d}/{num_requests_open:3d} dispatched and {num_requests_postponed:3d}/{num_requests_open:3d} postponed | Routes: {len(epoch_solution):2d} with cost {cost:6d}")

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, f"Reward should be negative cost of solution. Reward={reward}, cost={cost}"
        assert not info['error'], f"Environment error: {info['error']}"
        
        total_reward += reward

    if args.verbose:
        log(f"Cost of solution: {-total_reward}")

    return total_reward


def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default='greedy', help="Baseline strategy used to decide whether to dispatch routes")
    # Note: these arguments are only for convenience during development, during testing you should use controller.py
    parser.add_argument("--instance", help="Instance to solve")
    parser.add_argument("--instance_seed", type=int, default=1, help="Seed to use for the dynamic instance")
    parser.add_argument("--solver_seed", type=int, default=1, help="Seed to use for the solver")
    parser.add_argument("--solver_seed_dynamic", type=int, default=1, help="Seed to use for the dynamic solver")
    parser.add_argument("--static", action='store_true', help="Add this flag to solve the static variant of the problem (by default dynamic)")
    parser.add_argument("--epoch_tlim", type=int, default=120, help="Time limit per epoch")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Provide a specific directory to use as tmp directory (useful for debugging)")
    parser.add_argument("--verbose", action='store_true', help="Show verbose output")

    parser.add_argument("--nbGranular", type=int)
    parser.add_argument("--fractionGeneratedNearest", type=float)
    parser.add_argument("--fractionGeneratedFurthest", type=float)
    parser.add_argument("--fractionGeneratedSweep", type=float)
    parser.add_argument("--fractionGeneratedRandomly", type=float)
    parser.add_argument("--minSweepFillPercentage", type=int)
    parser.add_argument("--maxToleratedCapacityViolation", type=int)
    parser.add_argument("--maxToleratedTimeWarp", type=int)
    parser.add_argument("--initialTimeWarpPenalty", type=float)
    parser.add_argument("--penaltyBooster", type=float)
    parser.add_argument("--minimumPopulationSize", type=int)
    parser.add_argument("--generationSize", type=int)
    parser.add_argument("--nbElite", type=int)
    parser.add_argument("--nbClose", type=int)
    parser.add_argument("--targetFeasible", type=float)
    parser.add_argument("--repairProbability", type=int)
    parser.add_argument("--growNbGranularAfterNonImprovementIterations", type=int)
    parser.add_argument("--growNbGranularAfterIterations", type=int)
    parser.add_argument("--growNbGranularSize", type=int)
    parser.add_argument("--growPopulationAfterNonImprovementIterations", type=int)
    parser.add_argument("--growPopulationAfterIterations", type=int)
    parser.add_argument("--growPopulationSize", type=int)
    parser.add_argument("--intensificationProbabilityLS", type=int)
    parser.add_argument("--diversityWeight", type=float)
    parser.add_argument("--useSwapStarTW", type=int)
    parser.add_argument("--skipSwapStarDist", type=int)
    parser.add_argument("--circleSectorOverlapToleranceDegrees", type=int)
    parser.add_argument("--minCircleSectorSizeDegrees", type=int)
    parser.add_argument("--preprocessTimeWindows", type=int)
    parser.add_argument("--useDynamicParameters", type=int)
    parser.add_argument("--randomGenerator", type=int)
    
    parser.add_argument("--logTimeCost", action='store_true', help="Print (to stderr) output table of time at which each solution cost is achieved.")
    parser.add_argument("--pruneRoutes", action='store_true', help="Enable to prune routes consisting of optional clients.")
    parser.add_argument("--thresholdSchedule", type=str)

    args, unknown = parser.parse_known_args()

    if args.tmp_dir is None:
        # Generate random tmp directory
        args.tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        cleanup_tmp_dir = True
    else:
        # If tmp dir is manually provided, don't clean it up (for debugging)
        cleanup_tmp_dir = False

    try:
        if args.instance is not None:
            env = VRPEnvironment(seed=args.instance_seed, instance=tools.read_vrplib(args.instance), epoch_tlim=args.epoch_tlim, is_static=args.static)
        else:
            assert args.strategy != "oracle", "Oracle can not run with external controller"
            # Run within external controller
            env = ControllerEnvironment(sys.stdin, sys.stdout)

        # Make sure these parameters are not used by your solver
        args.instance = None
        args.instance_seed = None
        args.static = None
        args.epoch_tlim = None

        if args.strategy == 'oracle':
            run_oracle(args, env)
        else:
            run_baseline(args, env)

        if args.instance is not None:
            log(tools.json_dumps_np(env.final_solutions))
    finally:
        if cleanup_tmp_dir:
            tools.cleanup_tmp_dir(args.tmp_dir)