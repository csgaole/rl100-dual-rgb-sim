from pathlib import Path
import os,sys,json,torch,hydra,numpy as np,dill,random,time,gc,hashlib
from omegaconf import OmegaConf
run=Path(sys.argv[1]);out=run/'heldout_stages.json'
OmegaConf.register_new_resolver('eval',eval,replace=True)
cfg=OmegaConf.load(run/'.hydra/config.yaml')
p=torch.load(run/'checkpoints/latest.ckpt',map_location='cpu',pickle_module=dill)
online_dirs=sorted((run/'online_ft').glob('*'))
online=online_dirs[-1] if online_dirs else run/'online_ft/not_started'
stages=[('bc',None),('offline_selected',run/'best'),('offline_last',run/'last'),('online_selected',run/'online_best'),('online_last',online/'online_last'),('online_ema_selected',run/'online_best_ema'),('online_ema_last',online/'online_last_ema')]
results=json.loads(out.read_text()) if out.exists() else []
requested=set(os.environ.get('EVAL_STAGES','').split(','))-{''}
episodes=int(os.environ.get('EVAL_EPISODES','100'))
for stage,cp in stages:
 if requested and stage not in requested:continue
 files=[run/'checkpoints/latest.ckpt'] if cp is None else [cp/'model.pt',cp/'encoder.pt']
 fingerprint={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 if any(x['stage']==stage and x['episodes']==episodes and x.get('fingerprint')==fingerprint for x in results):
  print('HELDOUT_CACHED',stage,flush=True);continue
 if cp is not None:assert (cp/'model.pt').exists(),cp
 model=hydra.utils.instantiate(cfg.policy)
 model.load_state_dict(p['state_dicts']['model'])
 if cp is not None:
  model.model.load_state_dict(torch.load(cp/'model.pt',map_location='cpu'))
  model.obs_encoder.load_state_dict(torch.load(cp/'encoder.pt',map_location='cpu'))
 model.cuda().eval();model.use_aug=False
 if hasattr(model.obs_encoder,'force_stochastic'):model.obs_encoder.force_stochastic=False
 torch.manual_seed(900000);torch.cuda.manual_seed_all(900000);np.random.seed(900000);random.seed(900000)
 runner=hydra.utils.instantiate(cfg.task.env_runner,output_dir=str(run),eval_episodes=episodes)
 runner.eval_seed_base=900000
 metrics=runner.run(model)
 results=[x for x in results if x['stage']!=stage]
 results.append(dict(fingerprint=fingerprint,stage=stage,checkpoint=str(cp or run/'checkpoints/latest.ckpt'),episodes=episodes,seed_base=900000,success_rate=float(metrics['test_mean_score']),mean_return=float(metrics['mean_returns']),role='fixed heldout evaluation; never used to select checkpoints',updated_unix=time.time()))
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2));tmp.replace(out);print('HELDOUT_STAGE_RESULT',json.dumps(results[-1]),flush=True)
 runner.env.close();del runner,model;gc.collect();torch.cuda.empty_cache()
print('HELDOUT_REQUEST_COMPLETE' if requested else 'PIPELINE_COMPLETE',flush=True)
