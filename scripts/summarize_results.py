"""Recompute success totals without treating different studies as paired trials."""
from pathlib import Path
import csv,json,collections
r=Path(__file__).resolve().parents[1];groups=collections.defaultdict(lambda:[0,0,0])
for row in csv.DictReader((r/'results/test_results_v2.csv').open()):
 key=(row['family'],row['stage'],row['test_seed_base']);g=groups[key]
 g[0]+=int(row['successes']);g[1]+=int(row['episodes']);g[2]+=1
print(json.dumps([dict(family=k[0],stage=k[1],test_seed_base=k[2],successes=v[0],episodes=v[1],runs=v[2],success_percent=100*v[0]/v[1]) for k,v in groups.items()],ensure_ascii=False,indent=2))
