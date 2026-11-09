"""Write plan_small.json and plan_large.json. Each task: dict(kind, pars, seed, flips, sub)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
small, large = [], []
small += [dict(kind='cubic', pars=[3, 2], seed=s, flips=30000, sub=2) for s in range(4, 17)]
small += [dict(kind='cubic', pars=[5, 3], seed=s, flips=100000, sub=2) for s in range(5, 17)]
small += [dict(kind='cubic', pars=[5, 3], seed=s, flips=0, sub=4) for s in (1, 2, 3)]
small += [dict(kind='cubic', pars=[5, 3], seed=s, flips=100000, sub=4) for s in (1, 2, 3)]
large += [dict(kind='cubic', pars=[8, 5], seed=s, flips=400000, sub=2) for s in range(5, 17)]
# included because the probe fits: peak 4.4 GB at 4 threads
large += [dict(kind='cubic', pars=[8, 5], seed=1, flips=0, sub=4), dict(kind='cubic', pars=[8, 5], seed=1, flips=400000, sub=4)]
json.dump(small, open(os.path.join(here, 'plan_small.json'), 'w'), indent=0)
json.dump(large, open(os.path.join(here, 'plan_large.json'), 'w'), indent=0)
print(len(small), len(large))
