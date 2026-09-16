from pathlib import Path
import ast, numpy as np, torch, gym
from types import SimpleNamespace
from rl_100.unidpg.uni_ppo import compute_gae_per_env, BehaviorProximalPolicyOptimization
from rl_100.gym_util.multistep_wrapper import MultiStepWrapper
r=torch.tensor([[[1.],[2.]],[[3.],[4.]]]);v=torch.ones_like(r)*2;vn=torch.ones_like(r)*5
# Env0 truncates after first sample; env1 truly terminates at last sample.
done=torch.tensor([[[1.],[0.]],[[0.],[1.]]]);dw=torch.tensor([[[0.],[0.]],[[0.],[1.]]]);lengths=torch.tensor([[[2.],[4.]],[[4.],[1.]]])
a,t=compute_gae_per_env(r,done,dw,v,vn,.99,.95,4,executed_steps=lengths)
delta=r+.99**lengths*(1-dw)*vn-v
expected=delta.clone();expected[0,1]+= .99**4*.95*delta[1,1]
assert torch.allclose(a.reshape_as(r),expected)
class Tiny(gym.Env):
 observation_space=gym.spaces.Box(-100,100,(1,),dtype=np.float32)
 action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
 def reset(self):self.n=0;return np.array([0],dtype=np.float32)
 def step(self,a):self.n+=1;return np.array([self.n],dtype=np.float32),float(self.n),False,{}
w=MultiStepWrapper(Tiny(),2,4,max_episode_steps=6,reward_agg_method='discounted_sum',gamma=.99);w.reset()
o,r,d,i=w.step(np.zeros((4,1)));assert i['executed_steps']==4 and not d;assert abs(r-sum(.99**k*(k+1) for k in range(4)))<1e-6
o,r,d,i=w.step(np.zeros((4,1)));assert i['executed_steps']==2 and d and np.asarray(i['TimeLimit.truncated']).any();assert abs(r-(5+.99*6))<1e-6
class Dynamics:
 prediction_mode='last'
 def step(self,last,act,hist):
  assert last.shape==(3,2) and hist.shape==(3,2,2)
  return (last+1).numpy(),np.zeros((3,1),dtype=np.float32),None,None
seen=[]
def adv(h,a,*unused):seen.append(h.clone());return h.sum(1)
fake=SimpleNamespace(cfg=SimpleNamespace(n_obs_steps=2),_device='cpu',advantage_computation=adv)
BehaviorProximalPolicyOptimization.NStepValueEstimation(fake,torch.zeros((3,4)),torch.zeros((3,4,1)),Dynamics(),None,None,None,4)
assert len(seen)==4 and torch.equal(seen[2][0],torch.tensor([1.,1.,2.,2.]))
print('SEMANTICS_PASS: discounted rewards, variable durations, terminal/truncation, per-env GAE, rolling latent history',flush=True)
