"""Dictionary observations; preserve time/env axes until per-env GAE is computed."""
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,args,shape_info,device,wo_visual=False,env_num=None,steps_per_update=None):
        assert not wo_visual
        self.args=args;self.shape_info=shape_info;self.device=device
        self.env_num=env_num;self.steps_per_update=steps_per_update or args.batch_size
        self.wo_visual=False;self.use_imagin_robot=False
        self.reset()

    def reset(self):
        lead=(self.args.batch_size,) if self.env_num is None else (self.steps_per_update,self.env_num)
        self.obs={k:np.zeros(lead+tuple(s),dtype=np.float32) for k,s in self.shape_info['obs'].items()}
        self.next_obs={k:np.zeros_like(v) for k,v in self.obs.items()}
        self.action=np.zeros(lead+(self.args.num_inference_steps+1,)+tuple(self.shape_info['action']),dtype=np.float32)
        self.a_logprob=np.zeros(lead+(self.args.num_inference_steps,)+tuple(self.shape_info['action']),dtype=np.float32)
        for key in ('reward','done','dw'):setattr(self,key,np.zeros(lead+(1,),dtype=np.float32))
        self.count=0

    def store(self,obs,action,a_logprob,reward,next_obs,done,dw):
        assert self.count<len(self.action)
        for key in self.obs:
            self.obs[key][self.count]=obs[key]
            self.next_obs[key][self.count]=next_obs[key]
        self.action[self.count]=action;self.a_logprob[self.count]=a_logprob
        for key,value in [('reward',reward),('done',done),('dw',dw)]:
            getattr(self,key)[self.count]=np.asarray(value).reshape(getattr(self,key)[self.count].shape)
        self.count+=1

    def numpy_to_tensor(self):
        to=lambda x:torch.as_tensor(x[:self.count],dtype=torch.float32,device=self.device)
        return ({k:to(v) for k,v in self.obs.items()},to(self.action),to(self.a_logprob),to(self.reward),
                {k:to(v) for k,v in self.next_obs.items()},to(self.dw),to(self.done))

    def numpy_to_tensor_vec(self):
        assert self.env_num is not None
        return self.numpy_to_tensor()

    def load_flattened(self,source):
        assert source.env_num is not None and self.env_num is None
        flatten=lambda x:x[:source.count].reshape(-1,*x.shape[2:])
        self.obs={k:flatten(v) for k,v in source.obs.items()}
        self.next_obs={k:flatten(v) for k,v in source.next_obs.items()}
        for key in ('action','a_logprob','reward','done','dw'):setattr(self,key,flatten(getattr(source,key)))
        self.count=source.count*source.env_num

if __name__=='__main__':
    from types import SimpleNamespace
    args=SimpleNamespace(batch_size=4,num_inference_steps=2)
    shapes={'obs':{'image_corner2':(2,3,4,5),'image_behindGripper':(2,3,4,5),'agent_pos':(2,9)},'action':(4,4)}
    b=ReplayBuffer(args,shapes,'cpu',env_num=2,steps_per_update=2)
    for t in range(2):
        obs={k:np.stack([np.full(s,10*t+e) for e in range(2)]) for k,s in shapes['obs'].items()}
        nxt={k:v+100 for k,v in obs.items()}
        b.store(obs,np.zeros((2,3,4,4)),np.zeros((2,2,4,4)),np.array([t,t+1]),nxt,[False,True],[False,False])
    flat=ReplayBuffer(args,shapes,'cpu');flat.load_flattened(b)
    o,a,lp,r,n,dw,done=flat.numpy_to_tensor()
    assert o['image_behindGripper'][:,0,0,0,0].tolist()==[0,1,10,11]
    assert torch.equal(n['image_corner2'],o['image_corner2']+100)
    assert done[:,0].tolist()==[0,1,0,1] and dw.sum()==0
    assert tuple(a.shape)==(4,3,4,4)
    print('DUAL_BUFFER_TIME_ENV_TERMINAL_TEST_PASS')
