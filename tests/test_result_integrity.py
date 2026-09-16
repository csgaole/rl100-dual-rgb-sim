"""Cross-check independent aggregate CSV against the per-episode evidence."""
import csv,collections
from pathlib import Path
r=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((r/'results/episode_results_v2.csv').open()));totals=collections.defaultdict(lambda:[0,0]);keys=set()
for row in rows:
 key=(row['run'],row['stage'],row['seed']);assert key not in keys,key;keys.add(key)
 v=totals[key[:2]];v[0]+=int(row['success']);v[1]+=1
summaries=list(csv.DictReader((r/'results/test_results_v2.csv').open()))
assert len(rows)==3000 and len(summaries)==30
for row in summaries:
 assert totals[(row['run'],row['stage'])]==[int(row['successes']),int(row['episodes'])],row
 assert abs(100*int(row['successes'])/int(row['episodes'])-float(row['success_percent']))<1e-8
original=list(csv.DictReader((r/'results/test_results.csv').open()));assert len(original)==28
for stage,expected in [('bc',377),('offline_selected',373),('online_selected',374),('online_last',391)]:
 selected=[x for x in original if x['stage']==stage];assert sum(int(x['episodes']) for x in selected)==400
 assert sum(int(x['successes']) for x in selected)==expected
print('RESULT_INTEGRITY_PASS: 3000 episode records, 30 follow-up stages, 28 original stages')
