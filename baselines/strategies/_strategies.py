import numpy as np
from environment import State
import math
from itertools import compress
import sys

from tools import _filter_instance

def angle_diff(a, b, scale=2*math.pi):
    return (b - a + scale/2) % scale - scale/2

def abs_angle_diff(a, b, scale=2*math.pi):
    return abs(angle_diff(a, b, scale=scale))

def _fangle(observation: State,
            rng: np.random.Generator):
    return _angle(observation, rng, fixed_angle=15)

def _rangle(observation: State,
            rng: np.random.Generator):
    return _angle(observation, rng, must_go_ratio=4.5)

def _angle(observation: State,
           rng: np.random.Generator,
           min_optional_to_dispatch = 2,
           frac_optional_to_dispatch = 0.80, # TODO: tune
           fixed_angle = None, # in degrees
           must_go_ratio = None,
           n_angles_when_empty=0, # when n_must_go = 0, use this many "must_go" angles
           ):
    '''Simple heuristic dispatching a fraction of clients with smallest angular separation from the nearest must-go client.'''
    mask = np.copy(observation['must_dispatch'])
    n_must_go = sum(mask)
    if len(mask) <= min_optional_to_dispatch + 1: # no decisions to make
        return _greedy(observation, rng)
    if n_must_go == 0:
        if n_angles_when_empty == 0:
            return _lazy(observation, rng)
    optional = ~mask
    optional[0] = False # depot
    n_optional = sum(optional)
    if n_optional <= min_optional_to_dispatch:
        return _greedy(observation, rng)
    n_optional_to_dispatch = max([min_optional_to_dispatch, int(frac_optional_to_dispatch*n_optional)])
    if must_go_ratio is not None:
        n_optional_to_dispatch = int(must_go_ratio * n_must_go)
    if n_optional_to_dispatch >= n_optional:
        return _greedy(observation, rng)
    is_depot = np.copy(observation['is_depot'])
    candidates = 1 - mask - is_depot
    assert is_depot[0], "Assuming depot has index 0!"
    depot_x, depot_y = observation['coords'][0]
    angles = np.array([math.atan2(y - depot_y, x - depot_x) for x, y in observation['coords']])
    
    if n_must_go <= max([0,n_angles_when_empty/2]):
        angle0 = angles[1] if n_must_go == 0 else angles[mask][0]
        must_go_angles = [i * math.pi * 2 / n_angles_when_empty + angle0 for i in range(n_angles_when_empty)]
    else:
        must_go_angles = angles[mask]
    angle_diffs = np.array([min([abs_angle_diff(a, b) for b in must_go_angles]) for a in angles])
    if fixed_angle is not None:
        # Use a fixed angle to calculate optional dispatches
        cutoff = fixed_angle*math.pi/180
    else:
        # Use n_optional_to_dispatch to calculate optional dispatches
        cutoff = np.partition(angle_diffs[optional], n_optional_to_dispatch-1)[n_optional_to_dispatch-1]
    mask = (mask | (optional & (angle_diffs <= cutoff)))
    mask[0] = True
    # print(mask)
    return _filter_instance(observation, mask)

def _fdist(observation: State,
            rng: np.random.Generator):
    return _dist(observation, rng, fixed_pct=0.25)

def _rdist(observation: State,
            rng: np.random.Generator):
    return _dist(observation, rng, must_go_ratio=5.0)

def _dist(observation: State,
           rng: np.random.Generator,
           min_optional_to_dispatch = 2,
           frac_optional_to_dispatch = 0.80, # TODO: tune
           fixed_pct = None, # always dispatch when insertion distance is within this relative percentage
           must_go_ratio = None,
           n_angles_when_empty=0, # when n_must_go = 0, use this many "must_go" angles
           ):
    '''Simple heuristic dispatching a fraction of clients with smallest angular separation from the nearest must-go client.'''
    mask = np.copy(observation['must_dispatch'])
    matrix = observation['duration_matrix']
    n_must_go = sum(mask)
    if len(mask) <= min_optional_to_dispatch + 1: # no decisions to make
        return _greedy(observation, rng)
    if n_must_go == 0:
        if n_angles_when_empty == 0:
            return _lazy(observation, rng)
    optional = ~mask
    optional[0] = False # depot
    n_optional = sum(optional)
    if n_optional <= min_optional_to_dispatch:
        return _greedy(observation, rng)
    n_optional_to_dispatch = max([min_optional_to_dispatch, int(frac_optional_to_dispatch*n_optional)])
    if must_go_ratio is not None:
        n_optional_to_dispatch = int(must_go_ratio * n_must_go)
    if n_optional_to_dispatch >= n_optional:
        return _greedy(observation, rng)
    is_depot = np.copy(observation['is_depot'])
    candidates = 1 - mask - is_depot
    assert is_depot[0], "Assuming depot has index 0!"
    must_go_list = list(compress(range(len(mask)), mask))
    distance_diffs = np.array([min([(matrix[0][i] + matrix[i][j]) / matrix[0][j] - 1 for j in must_go_list]) for i in range(len(mask))])
    # sys.stderr.write(str(distance_diffs[optional]))
    if fixed_pct is not None:
        # Use a fixed angle to calculate optional dispatches
        cutoff = fixed_pct
    else:
        # Use n_optional_to_dispatch to calculate optional dispatches
        cutoff = np.partition(distance_diffs[optional], n_optional_to_dispatch-1)[n_optional_to_dispatch-1]
    mask = (mask | (optional & (distance_diffs <= cutoff)))
    mask[0] = True
    # print(mask)
    return _filter_instance(observation, mask)


def _greedy(observation: State, rng: np.random.Generator):
    mask = np.ones_like(observation['must_dispatch']).astype(np.bool8)
    return _filter_instance(observation, mask)


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


STRATEGIES = dict(
    angle=_angle,
    fangle=_fangle,
    rangle=_rangle,
    dist=_dist,
    fdist=_fdist,
    rdist=_rdist,
    greedy=_greedy,
    lazy=_lazy,
    random=_random
)
