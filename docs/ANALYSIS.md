# 成功率、根因与对策

## 1. 先区分三种问题

实现错误、统计/选择误差、算法能力不足是不同层次。修复实现后，没有显著收益仍可能来自高起点BC、数据覆盖不足、critic误差或评价噪声；不能继续把每个低分都解释为某一处代码bug。

## 2. 已有较强证据的实现问题

**EMA / BN buffer。** 共享图像encoder仍含BatchNorm，旧EMA只更新parameters，未同步running_mean/var等buffers。相同checkpoint修复60个BN buffer，seed100的20次验证由0%到100%，raw policy约95%。`ema_repaired.py`保留修复；首次收集和训练前必须检查raw/EMA在同输入下的输出。该诊断对特定checkpoint有很强因果证据，不证明所有RGB低分都来自BN。

**Offline Dropout likelihood。** 旧策略处于train、新策略eval，或两个train forward使用独立mask，会使同权重的概率比也偏离1。诊断中old-train/new-eval中位ratio=.204871，两个train=.034855；两者eval时10步均ratio=1、logprob差0。`train_offline_fixed.py`在生成旧策略和优化前显式eval；eval不关闭autograd，因此actor仍能更新。后续97/92表明这是必要正确性修复，但尚不足以带来稳定成功率提升。

**Online transition语义。** 保留true terminal与time-limit truncation的区别，截断允许bootstrap，episode边界停止GAE向下一episode传播；chunk奖励按gamma折扣，bootstrap按实际执行长度gamma^k；vector buffer按env/time维度对齐。`tests/test_semantics.py`检查这些边界以及latent history滚动，不能用仅“loss下降”替代。

**多相机buffer与渲染。** 原single `image`字段不能表示两路图像，改为按obs metadata保存，并修复flatten轴。每个MuJoCo context渲染前显式绑定framebuffer，初始化context后再reset目标site。不同环境切换时不这样处理会引入错误画面。

## 3. Offline 为什么收益很小

BC已94–95%，失败样本稀少，而训练数据全部是成功expert轨迹。Q可以区分完全随机动作，却未必能区分两个都接近expert的细微修正。Q训练的回归目标随bootstrap变化；训练loss不单调不是单凭曲线就能断言发散。

历史Q对expert与random偏好一致率约84%，对expert与小扰动约69%；这只是模型偏好，不是真实回报标签验证。初始仿真分支Q排序与后续return一致率86/168与78/168（约51%/46%），但该评测有折扣、继续策略、动作执行与图像差异等混杂。

进一步对齐实验每seed仅5个场景×2状态×3候选×3重复；对每状态重复取平均后30个pair。IQL与r+gammaV排序13/30、16/30；FQE与折扣return14/30、13/30。full_state全部相同，但图像hash仅28/90、34/90相同。哈希不同不一定代表大视觉误差；一次固定状态测试最大动作差仅约2e-4。重复分支来自少数场景，不能当作90个独立任务。

FQE held-out RMSE后期变差（约3.90→4.84、3.71→4.58），并且FQE继续策略每primitive step重规划，与部署4步chunk不同。因此不能把FQE当成无误差ground truth，也不能据现有样本断言critic毫无价值。

四步模型展开会叠加dynamics/critic误差：最终四步优势94/93，单步97/95。普通BC续训98/98已经追平优势加权98/99，进一步削弱“提升必然来自Q”的解释。

## 4. 统计与checkpoint选择

原始四RL运行：BC377/400，Offline selected373/400，Online selected374/400，Online last391/400。400条包含相同100个环境seed的多模型重复，以及共享BC初始化；不代表400个独立训练样本。Online last是训练完成后的辅助指标，不能见其更高就改变主要终点。

后续系列测试起点910000、920000、930000不同；同一BC在相同评估规范下重评仍可相差2–4pp。50次validation最小刻度2pp，100次test最小1pp；98/100与99/100相差1条episode，不足以确立算法优势。名义Wilson区间只描述episode层面二项不确定性，不消除场景相关和训练种子不足。

## 5. 推荐下一轮实验

1. 锁定环境版本、相机XML、renderer warm-up、目标site、checkpoint SHA、所有随机流；记录初始state/image/action。重复同checkpoint至少3次，先估计评估噪声。
2. 固定validation和未接触test场景，各方法共享场景，至少5个独立训练seed；验证≥50，正式测试≥300，报告配对失败变化及seed间分布。样本数不能取代正确的独立性设计。
3. 同预算比较原BC、普通BC续训、优势加权、修复BPPO；再分别比较BC→Online与BC→Offline→Online。当前续训新模型尚未进行配套Online，不可把其分数和旧Online串为同一条训练链。
4. 新增失败、接近插孔但未插入、恢复轨迹；必须保持训练/验证/测试场景隔离。只在成功窄分布上增大critic网络，未必补足动作排序监督。
5. 对齐4步执行的critic和验证return；分离成功率与dense reward，报告排序随动作差距变化、ensemble不确定性、校准误差。必要时限制Q权重，保留BC锚点。
6. 在相同数据、参数预算和评估协议下比较两视角独立/共享encoder、冻结/微调、分辨率或crop、视觉增强。不要同时换数据、模型、相机和优化器后仅归因于RGB表示。

## 6. 历史 Door 与单视角的边界

Adroit Door为灵巧手任务，不是平行夹爪Peg任务。早期200条×100步数据、RGB/PCD模型、Recon/VIB、timeout等配置与本双视角Peg系列不同。Door阶段包含中止、迁移、修复和共享GPU，archive只作为排查记录，未整理成同一最终公平对照表，不能用中途validation冒充最终held-out结论。

单视角Peg完整结果另见 `single_view_comparison.md`：PCD BC/Offline/Online selected均值84/84.5/93.5%，单RGB80/82/88%。它与双RGB不是只变相机数量的严格消融。
