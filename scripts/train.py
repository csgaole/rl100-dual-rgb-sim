"""Portable, explicit stage launcher. Existing output directories are never overwritten."""
import argparse, copy, json, os, pathlib, subprocess, sys
from omegaconf import OmegaConf
p=argparse.ArgumentParser(__doc__)
p.add_argument('stage',choices=['bc','offline','full'])
p.add_argument('--seed',type=int,default=100)
p.add_argument('--run',required=True)
p.add_argument('--bc-checkpoint',help='Trusted validation-selected BC checkpoint; required for RL')
p.add_argument('--bc-policy',choices=['model','ema_model'],default='ema_model')
p.add_argument('--eval-episodes',type=int,default=50)
p.add_argument('--historical-dropout',action='store_true',help='Reproduce historical unfixed BPPO; not recommended')
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1]
os.environ.update(RL100_CORE_ROOT=str(root/'rl100'),RL100_DUAL_ROOT=str(root/'experiments/dual_rgb'),RL100_CONFIG_ROOT=str(root/'configs/historical'))
config=root/'configs/historical'/('peg_dual_rgb_bc_seed100.yaml' if a.stage=='bc' else 'dual_rgb_rl_full_100.yaml')
c=OmegaConf.load(config);OmegaConf.set_struct(c,False)
run=pathlib.Path(a.run).resolve()
if run.exists():p.error('Run already exists; choose a new output directory')
if a.stage!='bc' and (not a.bc_checkpoint or not pathlib.Path(a.bc_checkpoint).is_file()):p.error('A valid --bc-checkpoint is required')
for role in ('dataset','critic_dataset','scale_dataset','finetune_dataset'):
 c.task[role].zarr_path=str(root/'experiments/dual_rgb/data/metaworld_peg_insert_dual_rgb_gain20.zarr')
 if not pathlib.Path(c.task[role].zarr_path).exists():p.error('Collect/prepare the demonstration dataset first')
c.training.seed=a.seed;c.training.resume=a.stage!='bc';c.training.num_epochs=250
c.ema._target_='ema_repaired.EMAModel';c.use_wandb=True
c.only_bc=a.stage=='bc';c.offline=a.stage!='bc';c.online=a.stage=='full';c.load_bc=False;c.stage1_only=False
c.exp_name=f'{a.stage}_seed{a.seed}';c.task.env_runner.eval_episodes=a.eval_episodes
c.unio4.stage1_resume_dir=str(run)
if a.smoke:
 c.training.num_epochs=1;c.training.max_train_steps=2;c.training.max_val_steps=1
 c.training.num_critic_epochs=1;c.dynamics.dynamics_max_epochs=1;c.dynamics.max_epochs_since_update=1
 c.unio4.bppo_steps=2;c.unio4.eval_freq=2;c.unio4.eval_step=2;c.unio4.finetune_batch_size=32
 c.ppo.max_train_steps=512;c.ppo.batch_size=128;c.ppo.mini_batch_size=32;c.ppo.K_epochs=2
 c.ppo.evaluate_freq=512;c.task.env_runner.eval_episodes=2
run.mkdir(parents=True);(run/'config').mkdir()
if a.stage!='bc':
 import torch,dill,hashlib
 checkpoint=pathlib.Path(a.bc_checkpoint).resolve();payload=torch.load(checkpoint,map_location='cpu',pickle_module=dill)
 payload['state_dicts']['model']=copy.deepcopy(payload['state_dicts'][a.bc_policy])
 payload['pickles']['_output_dir']=dill.dumps(str(run));payload['cfg']=c
 (run/'checkpoints').mkdir();torch.save(payload,run/'checkpoints/latest.ckpt',pickle_module=dill)
 (run/'bc_source.json').write_text(json.dumps({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'policy':a.bc_policy},indent=2))
OmegaConf.save(c,run/'config/experiment.yaml')
trainer='train_repaired.py' if a.stage=='bc' else ('train_dual_rl.py' if a.historical_dropout else 'train_offline_fixed.py')
cmd=[sys.executable,str(root/'experiments/dual_rgb'/trainer),'--config-path='+str(run/'config'),'--config-name=experiment','hydra.run.dir='+str(run)]
(run/'launch.json').write_text(json.dumps({'stage':a.stage,'smoke':a.smoke,'command':cmd,'note':'fixed dropout and 50 validation episodes are new defaults, not historical scores'},indent=2))
raise SystemExit(subprocess.call(cmd))
