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
    solutions = []
    executable = os.path.join('baselines', 'vroom', 'bin', 'vroom')
    instance_filename_json = os.path.join(tmp_dir, "problem.json")
    tools.write_json(instance_filename_json, instance, steps=init_routes)
    with subprocess.Popen([
                executable, '-i', instance_filename_json,
                '-l', str(time_limit),
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
                solutions.append((cost, routes))
        except:
            pass
    
    return solutions

def solve_static_vrptw(instance, time_limit=3600, tmp_dir="tmp", seed=1, args=None):
    time_limit = time_limit
    start_time = time.time()
    vroom_timelimit = int(min(time_limit * args.warmstartTimeFraction, args.maxWarmstartTime))
    best_cost = float('inf')
    time_cost = [] # stores all solution costs and the time at which they were found
    
    do_vroom_warmstart = vroom_timelimit > MIN_VROOM_TIME
    
    hgs_warmstart_num_trials = args.nbHgsWarmstarts
    hgs_warmstart_mode = args.hgsWarmstartMode
    hgs_warmstart_time = args.hgsWarmstartTime
    do_hgs_warmstart = hgs_warmstart_num_trials > 0 and hgs_warmstart_time >= MIN_HGS_TIME
    
    rng=np.random.default_rng(args.solver_seed)

    costs = [] # stores all vroom solution costs
    # Prevent passing empty instances to the static solver, e.g. when
    # strategy decides to not dispatch any requests for the current epoch
    if instance['coords'].shape[0] <= 1:
        yield [], 0
        return

    if instance['coords'].shape[0] <= 2:
        solution = [[1]]
        cost = tools.validate_static_solution(instance, solution)
        yield solution, cost
        return

    os.makedirs(tmp_dir, exist_ok=True)

    warmstart_filepath = os.path.join(tmp_dir, "warmstart")
    fout = None

    if do_vroom_warmstart:
        # Call VROOM solver
        executable = os.path.join('baselines', 'vroom', 'bin', 'vroom')
        # executable = tools.which('vroom')
        assert executable is not None, f"VROOM executable {executable} does not exist!"

        init_routes = []
        init_routes = [[185, 24, 98, 33, 246, 74, 1, 170, 115, 55, 31, 2], [17, 65, 100, 109, 108, 113, 125, 140, 141, 253, 120, 232, 23, 142, 15, 111, 243, 61, 167], [47, 164, 93, 139, 11, 18, 228, 7, 257, 188, 9, 245, 119, 39, 37, 48, 49, 225, 78, 179, 151, 171], [85, 200, 191, 5, 58, 254, 176, 114, 56, 138, 13, 124, 103, 241, 3, 51, 133, 106, 30], [233, 81, 117, 46, 42, 43, 16, 136, 45, 189, 247, 69, 172, 70, 194, 54, 20, 90, 53, 122, 237, 229], [236, 207, 248, 59, 231, 175, 131, 79, 215, 38, 216, 68, 4, 77, 252, 166, 112, 239, 26, 118, 64], [177, 226, 214, 187, 123, 222, 84, 75, 255, 110, 62, 96, 19, 258, 63, 71, 34, 87, 92], [10, 134, 73, 83, 22, 155, 163, 235, 127, 40, 234, 116, 44, 198, 238, 240, 143, 76, 91, 199, 14], [154, 72, 94, 82, 182, 165, 126, 132, 89, 173, 32, 251, 25, 244, 80, 203, 190, 227, 99, 144, 27, 52], [230, 156, 256, 129, 135, 57, 60, 8, 184, 197, 195, 178, 29, 28, 12, 105, 212, 242, 183, 97, 88, 169, 204, 218], [223, 213, 221, 211, 202, 210, 209, 208, 224, 157, 160, 217, 149, 150, 148, 181, 193, 86, 201, 102, 220, 219, 206, 205, 196, 180], [50, 35, 147, 152, 161, 158, 153, 130, 107, 249, 250, 101, 128, 121, 146, 145, 159, 162], [174, 67, 104, 66, 168, 6, 36, 186, 95, 137, 41, 21, 192]]
        fout = open(warmstart_filepath, "w")    # this file stores are vroom solutions to get fed into HGS
        instancetmp = copy.deepcopy(instance)   # this is an instance whose duration matrix gets updated so vroom produces new results
        time_remaining = vroom_timelimit - (time.time() - start_time)

        solutions = run_vroom(instancetmp, tmp_dir, time_remaining, args.exploreLevel, init_routes=init_routes)
        costs = [s[0] for s in solutions]
        if len(solutions) > 0:
            cost, routes = min(solutions)
            if cost < best_cost:
                best_cost = cost
                yield routes, cost
                time_cost.append((time.time()-start_time, cost))
            
        # String to output to warmstart file
        for cost, routes in solutions:
            fout.write(routesToStr(routes) + "\n")
            
        time_remaining = vroom_timelimit - (time.time() - start_time)
    
        if args.verbose:    
            time_left = time_limit - (time.time()-start_time)
            print(f"\nVROOM found {len(costs)} solutions in {round(time.time()-start_time,1)} seconds. The costs are {costs}. Remaining time: {time_left}", file=sys.__stderr__)
    
    instance_filename = os.path.join(tmp_dir, "problem.vrptw")
    tools.write_vrplib(instance_filename, instance, is_vrptw=True)
    executable = os.path.join('baselines', 'hgs_vrptw', 'genvrp')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    
    
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    if len(costs) > 0:
        argList = [ executable, instance_filename, str(time_limit), 
                    '-veh', '-1', '-useWallClockTime', '1',
                    '-warmstartFilePath', warmstart_filepath
                ]
    else:
        argList = [ executable, instance_filename, str(time_limit), 
                    '-veh', '-1', '-useWallClockTime', '1'
                ]
    
    # Add all the tunable HGS args
    if args is not None:
        vargs = vars(args)
        for hgs_arg in ALL_HGS_ARGS:
            if hgs_arg in vargs and vargs[hgs_arg] is not None:
                argList += [f'-{hgs_arg}', str(vargs[hgs_arg])]
    
    if do_hgs_warmstart:
        hgs_warmstart_start = time.time()
        if fout is None:
            fout = open(warmstart_filepath, "w")    # this file stores are vroom solutions to get fed into HGS
        
        # Add the timeout
        hgs_warmstart_argList = [ 'timeout', str(hgs_warmstart_time)] + argList
        
        hgs_warmstart_results = [] # stores all hgs warmstart solutions
        hgs_warmstart_best_cost = float("inf")
        hgs_warmstart_worst_cost = float("-inf")
        
        for hgs_warmstart_trial in range(hgs_warmstart_num_trials):
            hgs_warmstart_seed = seed + 1 + hgs_warmstart_trial
            trial_best_cost = float("inf")
            
            # Add all the solver seed
            hgs_warmstart_argList_trial = hgs_warmstart_argList + ['-seed', str(hgs_warmstart_seed)]
            
            with subprocess.Popen(hgs_warmstart_argList_trial, stdout=subprocess.PIPE, text=True) as p:
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
                        if cost < trial_best_cost:
                            trial_best_cost = cost
                            if cost < best_cost:
                                check_cost = tools.validate_static_solution(instance, solution)
                                assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                                yield solution, cost
                                best_cost = cost
                                time_cost.append((time.time()-start_time, cost))
                        # Start next solution
                        routes = []
                    elif "EXCEPTION" in line:
                        raise Exception("HGS failed with exception: " + line)
                assert len(routes) == 0, "HGS has terminated with incomplete solution (is the line with Cost missing?)"
                if trial_best_cost < float("inf"):
                    hgs_warmstart_best_cost = min([trial_best_cost, hgs_warmstart_best_cost])
                    hgs_warmstart_worst_cost = max([trial_best_cost, hgs_warmstart_worst_cost])
                    result = dict({'cost': trial_best_cost, 'routes':solution})
                    hgs_warmstart_results.append(result)
            
        # write warmstart results to file
        found_routes = []
        hgs_warmstart_costs = []
        for result in hgs_warmstart_results:
            cost = result['cost']
            hgs_warmstart_costs.append(cost)
            if hgs_warmstart_mode=="BEST" and cost !=  hgs_warmstart_best_cost:
                continue
            elif hgs_warmstart_mode=="BEST_WORST" and cost !=  hgs_warmstart_best_cost and cost !=  hgs_warmstart_worst_cost:
                continue
            routes = result['routes']
            # String to output to warmstart file
            routeStr = ",".join(str(v) for v in routes[0])
            for route in routes[1:]:
                routeStr += "~"
                routeStr += ",".join(str(v) for v in route)
            fout.write(routeStr + "\n")
            found_routes.append(len(routes))
            
        if args.verbose:    
            time_left = time_limit - (time.time()-start_time)
            print(f"\nhgs warmstart found {len(hgs_warmstart_results)} solutions in {round(time.time()-hgs_warmstart_start,1)} seconds. The costs are {hgs_warmstart_costs}. Remaining time: {time_left}", file=sys.__stderr__)
    
    if fout is not None:
        fout.close()

    hgs_timelimit = max(time_limit - (time.time()-start_time), MIN_HGS_TIME)
    hgs_max_time = int(hgs_timelimit+1)
    
    
    # Add all the solver seed
    argList += ['-seed', str(seed)]
    
    # Add the timeout
    argList = [ 'timeout', str(hgs_max_time)] + argList

    hgs_start = time.time()
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
                if cost < best_cost:
                    check_cost = tools.validate_static_solution(instance, solution)
                    assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                    yield solution, cost
                    best_cost = cost
                    time_cost.append((time.time()-start_time, cost))
                # Start next solution
                routes = []
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with incomplete solution (is the line with Cost missing?)"
        
    if args.verbose:
        log(f"hgs found {len(time_cost)} solutions in {time.time() - hgs_start} seconds.")
    if 'logTimeCost' in args and args.logTimeCost:
        log("time\tcost")
        for row in time_cost:
            log(f"{row[0]}\t{row[1]}")
        

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
            if n_cust < 300:
                args.epoch_tlim = 3*60
            elif n_cust < 500:
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
            solutions = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, tmp_dir=args.tmp_dir, seed=args.solver_seed, args=args))
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