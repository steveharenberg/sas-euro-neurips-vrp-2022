# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import subprocess
import sys
import os
import json
import uuid
import platform
import numpy as np
import time
import copy
import itertools


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

MIN_VROOM_TIME = 1
MIN_HGS_TIME = 0.5

def routesToStr(routes):
    if len(routes) == 0:
        return ""

    routesStr = ",".join(str(v) for v in routes[0])
    for route in routes[1:]:
        routesStr += "~"
        routesStr += ",".join(str(v) for v in route)    
    return routesStr


def run_vroom(instance, tmp_dir, time_limit, explore_level, init_routes=[]):
    if args.verbose:
        print(f"\nRunning VROOM", file=sys.__stderr__)
    start_time = time.time()
    solutions = []
    executable = os.path.join('baselines', 'vroom', 'bin', 'vroom')
    instance_filename_json = os.path.join(tmp_dir, "problem_vroom.json")
    tools.write_json(instance_filename_json, instance, steps=init_routes)

    with subprocess.Popen([
                'timeout', str(time_limit),
                executable, '-i', instance_filename_json,
                '-x', str(explore_level),
                '-t', '1', # single threaded
                '-s'
            ], stdout=subprocess.PIPE, text=True) as p:
        try:    # hopefully this prevents issues if vroom finds no solution: we just move to HGS
            for line in p.stdout:
                line = line.strip()
                vroom_output = json.loads(line)
                cost = int(vroom_output["summary"]["cost"])
                routes = []
                for route in vroom_output["routes"]:
                    routes.append([int(node["location_index"]) for node in route["steps"] if node["type"]=="job"])
                solutions.append((routes, cost))
        except:
            pass
    if args.verbose:
        print(f"VROOM found {len(solutions)} solutions in {round(time.time()-start_time,1)} seconds. The costs are {[s[1] for s in solutions]}\n", file=sys.__stderr__)
    
    return solutions


def run_hgs(instance_filename, warmstart_filepath, time_limit=3600, tmp_dir="tmp", seed=1, num_sols_ignore=0, min_improvement=0, min_sols=0, args=None):
    if args.verbose:    
        print(f"\nRunning HGS", file=sys.__stderr__)
    executable = os.path.join('baselines', 'hgs_vrptw', 'genvrp')    
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"

    argList = [ executable, instance_filename, str(time_limit), 
                '-veh', '-1', '-useWallClockTime', '1'
              ]
    if warmstart_filepath is not None:
        argList += ['-warmstartFilePath', warmstart_filepath]
        
    if args is not None:
        vargs = vars(args)
        for hgs_arg in ALL_HGS_ARGS:
            if hgs_arg in vargs and vargs[hgs_arg] is not None:
                argList += [f'-{hgs_arg}', str(vargs[hgs_arg])]    
    argList += ['-seed', str(seed)]
    argList = [ 'timeout', str(time_limit)] + argList

    nsols = 0
    solutions = []
    
    # log(' '.join(argList))
    with subprocess.Popen(argList, stdout=subprocess.PIPE, text=True) as p:
        routes = []
        for line in p.stdout:
            line = line.strip()
            # Parse only lines which contain a route
            if line.startswith('Route'):
                if nsols >= num_sols_ignore:
                    label, route = line.split(": ")
                    route_nr = int(label.split("#")[-1])
                    assert route_nr == len(routes) + 1, "Route number should be strictly increasing"
                    routes.append([int(node) for node in route.split(" ")])
            elif line.startswith('Cost'):
                # End of solution
<<<<<<< HEAD
                nsols += 1
                if nsols > num_sols_ignore:
                    solution = routes
                    cost = int(line.split(" ")[-1].strip())
                    improvement = float('inf')
                    if len(solutions) > 0:
                        improvement = float(solutions[-1][1]) / cost
                    if len(solutions) == 0 or cost < solutions[-1][1]:
                        solutions.append((solution, cost))
                    # Start next solution
                    routes = []
                    if args.verbose:
                        print(f"cost: {cost}, improvement: {improvement}", file=sys.__stderr__)   
                    if improvement < min_improvement and nsols > num_sols_ignore and nsols - num_sols_ignore > min_sols:
                        break                 
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with incomplete solution (is the line with Cost missing?)"
    
    return solutions    

def get_subproblem(instance, solution, rng, k = 3, route_id = None):
    n_routes = len(solution)
    if k >= n_routes:
        return instance, list(range(n_routes))
    if route_id is None:
        route_id = rng.integers(n_routes)
    # find centroids
    coords = instance['coords']
    centroids = [np.mean(coords[route], axis=0) for route in solution]
    # log(centroids)
    # find nearest neighbors
    centroid_ = centroids[route_id]
    distances = [np.sum(np.square(centroid - centroid_)) for centroid in centroids]
    # log(distances)

    subproblem_routes = sorted(range(len(distances)), key=lambda x: distances[x])[:k] 
    # log('')
    # log(subproblem_routes)

    subproblem_clients = list(itertools.chain.from_iterable([solution[r] for r in subproblem_routes]))
    # log('')
    # log(subproblem_clients)

    mask = np.zeros_like(instance['must_dispatch']).astype(np.bool8)
    mask[0] = True
    mask[subproblem_clients] = True
    # log(tools.filter_instance(instance, mask))
    return tools.filter_instance(instance, mask), subproblem_routes

def subproblem_warm_start(routes, warmstart_filepath):
    # yield routes, cost
    if warmstart_filepath is not None:
        with open(warmstart_filepath, 'w') as fout:
            fout.write(routesToStr(routes) + "\n")

def solve_static_vrptw(instance, info, time_limit=3600, tmp_dir="tmp", seed=1, rng=None, subproblem_min_k=7, subproblem_time_limit=5, initial_time=5, warmstart=False, args=None):
    start_time = time.time()
    if rng is None:
        rng = np.random.default_rng(args.solver_seed)

    os.makedirs(tmp_dir, exist_ok=True)
    instance_filename = os.path.join(tmp_dir, "problem.vrptw")
    tools.write_vrplib(instance_filename, instance, is_vrptw=True)    
    warmstart_filepath = os.path.join(tmp_dir, "warmstart") if warmstart else None
    args.warmstartFilePath = warmstart_filepath
    
    curr_solutions = []
    remain_time = time_limit - (time.time() - start_time)
    iter_time = int(min([initial_time, remain_time]))
    iteration = 0
    
    curr_solutions += run_hgs(instance_filename, None, iter_time, tmp_dir, seed, 0, 1.001,  1, args)
    routes, cost = curr_solutions[-1]
    yield routes, cost
    
    
    remain_time = time_limit - (time.time() - start_time)
    while remain_time > 1:
        iteration += 1
        iter_time = int(min([subproblem_time_limit, remain_time]))
        subproblem_max_k = max([len(routes)//3, subproblem_min_k])
        subproblem_k = rng.integers(subproblem_max_k - subproblem_min_k) + subproblem_min_k if subproblem_max_k > subproblem_min_k else subproblem_min_k
        sub_instance, sub_instance_routes = get_subproblem(instance, routes, rng, k=subproblem_k)
        sub_instance_filename = os.path.join(tmp_dir, "subproblem.vrptw")
        tools.write_vrplib(sub_instance_filename, sub_instance, is_vrptw=True)
        
        # info_map = dict(zip(sub_instance['request_idx'], instance['request_idx'][list(range(len(sub_instance['request_idx'])))])   )
        # log(f"Running hgs for {iter_time} seconds. Remaining time: {remain_time} seconds.")
        # log([routes[r] for r in sub_instance_routes])
        if warmstart:
            assert info['is_static'], "subproblem warmstart not yet supported for dynamic"
            inv_map = dict(zip(sub_instance['request_idx'], instance['request_idx'][list(range(len(sub_instance['request_idx'])))])   )
            subproblem_warm_start([[inv_map[k] for k in routes[r]] for r in sub_instance_routes], warmstart_filepath)
            
        sub_solutions = run_hgs(sub_instance_filename, warmstart_filepath, iter_time, tmp_dir, seed + iteration, 0, 1.000, 0, args)
        if len(sub_solutions) > 0:
            sub_routes, sub_cost = sub_solutions[-1]
            new_routes = [i for j, i in enumerate(routes) if j not in sub_instance_routes]
            new_routes += [instance['request_idx'][route] for route in sub_routes]
            # log(f"Original sub: {[routes[r] for r in sub_instance_routes]}, New sub sol: {sub_routes}")
            new_cost = tools.validate_static_solution(instance, new_routes, allow_skipped_customers=not info['is_static'])
            # log(f"Original cost: {cost}, New cost: {new_cost}, Sub cost: {sub_cost}")
            if new_cost < cost:
                curr_solutions.append((new_routes, new_cost))
                routes, cost = curr_solutions[-1]
                yield routes, cost
        remain_time = time_limit - (time.time() - start_time)

    

# def solve_static_vrptw2(instance, time_limit=3600, tmp_dir="tmp", seed=1, args=None):   
#     os.makedirs(tmp_dir, exist_ok=True)
#     instance_filename = os.path.join(tmp_dir, "problem.vrptw")
#     tools.write_vrplib(instance_filename, instance, is_vrptw=True)    
#     warmstart_filepath = os.path.join(tmp_dir, "warmstart")
#     args.warmstartFilePath = warmstart_filepath
    
#     run_hgs(instance_filename, warmstart_filepath, int(time_limit), tmp_dir, seed, 0, 0, args)
#     for routes, cost in curr_solutions:
#         yield routes, cost     

  
        

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
    oracle_solution = min(solve_static_vrptw(hindsight_problem, info, time_limit=epoch_tlim, tmp_dir=args.tmp_dir), key=lambda x: x[1])[0]
    oracle_cost = tools.validate_static_solution(hindsight_problem, oracle_solution)
    log(f"Found oracle solution with cost {oracle_cost}")

    total_reward = run_baseline(args, env, oracle_solution=oracle_solution)
    assert -total_reward == oracle_cost, "Oracle solution does not match cost according to environment"
    return total_reward


def run_baseline(args, env, oracle_solution=None):

    rng = np.random.default_rng(args.solver_seed)

    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    observation, static_info = env.reset()
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
        epoch_tlim = args.epoch_tlim
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
            epoch_instance_dispatch = STRATEGIES[args.strategy](epoch_instance, rng)

            # Run HGS with time limit and get last solution (= best solution found)
            # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
            # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
            solutions = list(solve_static_vrptw(epoch_instance_dispatch, static_info, time_limit=epoch_tlim, tmp_dir=args.tmp_dir, seed=args.solver_seed, rng=rng, args=args))
            assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
            epoch_solution, cost = solutions[-1]

            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]
        
        if args.verbose:
            num_requests_dispatched = sum([len(route) for route in epoch_solution])
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_requests_postponed = num_requests_open - num_requests_dispatched
            log(f" {num_requests_dispatched:3d}/{num_requests_open:3d} dispatched and {num_requests_postponed:3d}/{num_requests_open:3d} postponed | Routes: {len(epoch_solution):2d} with cost {cost:6d}")

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, f"Reward should be negative cost of solution {reward}!={-cost}"
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
    # vroom warmstart
    parser.add_argument("--exploreLevel", type=int, default=0)
    parser.add_argument("--warmstartTimeFraction", type=float, default=0.0)
    parser.add_argument("--maxWarmstartTime", type=float, default=float('inf'))
    # hgs warmstart
    parser.add_argument("--nbHgsWarmstarts", type=int, default=0)
    parser.add_argument("--hgsWarmstartTime", type=float, default=2)
    parser.add_argument("--hgsWarmstartMode", choices=['BEST', 'BEST_WORST', 'ALL'], default='BEST')
    # other
    parser.add_argument("--logTimeCost", action='store_true')
    parser.add_argument("--randomGenerator", type=int)
    
    parser.add_argument("--logTimeCost", action='store_true', help="Print (to stderr) output table of time at which each solution cost is achieved.")

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