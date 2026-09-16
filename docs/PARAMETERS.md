# 参数：配置值、实际行为与计数单位

权威输入是 `configs/historical/` 中的完整 YAML。不要仅依据某个字段名称推断行为：该代码有多个实验分支，未启用分支的字段仍会出现在文件里。

## BC 与数据

双 RGB、84→224、两帧历史、9D proprioception、4D action、horizon=5、execute=4、DDIM=10。共享 R3M ResNet18；`use_pretrained_2DEncoder=false` 不等于不使用 R3M，实际 `encoders.resnet.rgb_model.weights=r3m`。

`use_group_norm=true` 在当时共享 RGB 分支没有真正替换 BatchNorm。因此历史模型仍含 BN；修改成真正 GroupNorm 会改变网络和 checkpoint 兼容性，不是原训练的同义配置。当前修复是 EMA 同步 buffers。

BC batch128，250 epochs，AdamW lr2e-4，betas(.95,.999)，eps1e-8，weight_decay1e-6，cosine，warmup500，grad accumulation1。每50 epochs rollout/checkpoint，每10 epochs验证 loss。EMA inv_gamma1/power.75/max.9999。原始运行选 epoch200；不是按照 held-out 测试分数选出的。

100×200=20,000 primitive transitions；90/10 episode 划分，seed42。dataset padding使窗口数不等同于原始 transition 数；约141训练 batch/epoch，窗口会重叠。`next_obs` 与后续观测逐步配对，末端保留 timeout；expert轨迹没有真正环境终止不意味着没有episode边界。

## Offline

Q/V：100 epochs，batch128；512×3 MLP，double Q，expectile .7，gamma .99，Q/V lr1e-4，target update频率2、tau.005；共享并冻结编码器。

Dynamics：7模型5elite、400×4、lr3e-4、batch128、patience10、configured max_epochs100。历史代码按epoch索引判断停止，精确执行数量以日志为准。

BPPO：2,000外层迭代，实际`finetune_batch_size=128`，10去噪步各更新一次，约20,000actor optimizer calls；lr5e-6线性退火，clip.25，早期clip decay.96，grad clip.5，entropy0，固定encoder，`use_gae=false`。`bppo_batch_size=512`是残留配置，不是这里真正采样128的替代值。

初始与每500迭代做validation；原始20次，后续50次；strict greater更新best。初始BC20/20时，新checkpoint即使20/20也不会替换它。best与last必须分别报告。

单步与四步优势消融仅改变优势估计的预测展开长度，部署仍执行4步。它不等于把动作维度从4改成16的chunk critic。单步IQL分支内部n_action_steps=1，因此该分支的gamma¹是预期行为，并未证实gamma⁴错误。

## Online

4env，每个决策执行4primitive actions。batch512chunks、minibatch128chunks、K_epochs3、10denoising steps：每完整轮4×3×10=120actor更新，4×3=12value更新。1M预算含488完整轮×2048=999,424steps，剩576未构成完整优化batch。总计58,560actor和5,856value更新，不含BC/Offline，也不把validation rollouts计入1M训练预算。

每个episode最多200steps，1M/200≈5,000个满长度episode等价量；实际episode计数取决于终止、截断和vector环境边界，不能只凭除法宣称精确完成5,000条独立episode。

lr_actor3e-6，lr_critic3e-4，gamma.99、GAE lambda.95、clip.2、adv norm、gradient clip、冻结encoder；每20k交互左右验证，原始20episode。`K_epochs=3`是每个rollout batch复用3遍，不是将历史dataset训练3个epoch。

## 续训对照与种子

普通BC与优势加权各2,000次Adam更新，batch128，lr5e-6线性下降，gradclip.5、encoder frozen、eval mode关闭Dropout。两组计算预算相同，但与BPPO约20k actor更新不是同预算对照。

Weighted：A=Q−V取演示第一条执行动作；u=exp(clamp(A,-5,ln10))，w=.5+.5u/mean(u)。w作用整个4步动作diffusion loss。beta=1 Q单位，非概率。加权并不保证Q排序正确。

训练seed100与300是不同随机初始化/采样流；评估环境seed单独指定。原始扩展seed200沿用100的BC，400沿用300的BC，所以四个RL种子只有两套独立BC初始化。
