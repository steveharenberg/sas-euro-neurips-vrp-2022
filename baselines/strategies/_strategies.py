import numpy as np
from environment import State
import math


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res

def angle_diff(a, b, scale=2*math.pi):
    return (b - a + scale/2) % scale - scale/2

def abs_angle_diff(a, b, scale=2*math.pi):
    return abs(angle_diff(a, b, scale=scale))

def _angle(observation: State, rng: np.random.Generator):
    '''Simple heuristic dispatching a fraction of clients with smallest angular separation from the nearest must-go client.'''
    mask = np.copy(observation['must_dispatch'])
    n_must_go = sum(mask)
    if n_must_go == 0:
        mask[1]=True # arbitrarily pick one
    optional = ~mask
    optional[0] = False # depot
    min_optional_to_dispatch = 2
    frac_optional_to_dispatch = 0.80 # TODO: tune
    n_optional = sum(optional)
    if n_optional <= min_optional_to_dispatch:
        return _greedy(observation, rng)
    n_optional_to_dispatch = max([min_optional_to_dispatch, int(frac_optional_to_dispatch*n_optional)])
    n_optional_to_dispatch = min([n_optional_to_dispatch, n_optional])
    is_depot = np.copy(observation['is_depot'])
    candidates = 1 - mask - is_depot
    assert is_depot[0], "Assuming depot has index 0!"
    depot_x, depot_y = observation['coords'][0]
    angles = np.array([math.atan2(y - depot_y, x - depot_x) for x, y in observation['coords']])
    must_go_angles = angles[mask]
    angle_diffs = np.array([min([abs_angle_diff(a, b) for b in must_go_angles]) for a in angles])
    cutoff = np.partition(angle_diffs[optional], n_optional_to_dispatch-1)[n_optional_to_dispatch-1]
    mask = (mask | (optional & (angle_diffs <= cutoff)))
    mask[0] = True
    # print(mask)
    return _filter_instance(observation, mask)


def _greedy(observation: State, rng: np.random.Generator):
    return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }


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
    greedy=_greedy,
    lazy=_lazy,
    random=_random
)
