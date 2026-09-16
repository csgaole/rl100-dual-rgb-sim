from dual_dataset import MetaworldMultiViewDataset

class Dataset(MetaworldMultiViewDataset):
    def get_shape_info(self,n_action_steps,n_obs_steps):
        sample=self._sample_to_data(self.sampler.sample_sequence(0))
        obs={}
        for key,value in sample['obs'].items():
            shape=tuple(value.shape[1:])
            if key.startswith('image_') and shape[-1]==3:
                shape=(shape[-1],)+shape[:-1]
            obs[key]=(n_obs_steps,)+shape
        return {'obs':obs,'action':(n_action_steps,sample['action'].shape[-1])}
