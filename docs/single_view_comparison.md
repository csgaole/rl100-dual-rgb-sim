# RL-100 Peg-Insert-Side: point-cloud vs single-camera RGB

Completed 2026-09-14 on `<TRAINING_HOST>`.

## Experiment

- Task: MetaWorld `peg-insert-side-v2`.
- Demonstrations: 100 successful paired trajectories, 200 steps each; 90 train and 10 validation episodes.
- Observations: either 512 XYZ points from the `corner2` depth render, or one `corner2` RGB image at 84×84; both use two observation frames and the same 9-D robot state.
- Policy: diffusion policy, horizon 5, executing 4 actions per decision.
- Offline RL: IQL value/critic training, learned dynamics, then 2,000 BPPO policy updates.
- Online RL: PPO from the completed offline checkpoint, 1,000,000 actual simulator transitions per seed, four environments, 2,048 transitions per update, three PPO epochs.
- Seeds: 100 and 300 for RL. Both use the same BC checkpoint, so the BC rows are repeated evaluations rather than two independently trained BC policies.
- Final evaluation: fixed held-out seed base 900000, 100 episodes for every stage and seed. Held-out results were never used to choose checkpoints.

## Primary held-out success rates

The primary online result is `online_selected`, selected by validation before held-out evaluation.

| Observation | RL seed | BC | Offline selected | Online selected |
|---|---:|---:|---:|---:|
| Point cloud | 100 | 84% | 86% | 94% |
| Point cloud | 300 | 84% | 83% | 93% |
| RGB, one camera | 100 | 81% | 83% | 82% |
| RGB, one camera | 300 | 79% | 81% | 94% |

Across the two 100-episode evaluations, point cloud averages 84.0% BC, 84.5% offline, and 93.5% online. RGB averages 80.0% BC, 82.0% offline, and 88.0% online. Pooled Wilson 95% intervals are 89.2–96.2% for point-cloud online and 82.8–91.8% for RGB online. Because only two RL seeds were run and RGB ranges from 82% to 94%, the 5.5-point mean gap is suggestive rather than a stable estimate of modality superiority.

## Auxiliary final-checkpoint results

These were evaluated for diagnosis and were not used to redefine the primary result.

| Observation | Seed | Online last | Online EMA selected | Online EMA last |
|---|---:|---:|---:|---:|
| Point cloud | 100 | 98% | 93% | 98% |
| Point cloud | 300 | 98% | 97% | 99% |
| RGB | 100 | 86% | 81% | 87% |
| RGB | 300 | 93% | 85% | 94% |

## Interpretation

Online PPO clearly improves point-cloud performance: 84.5% offline to 93.5% online on average. RGB improves from 82.0% to 88.0% on average, but the seed spread is large. Seed 300 reaches 94%, while seed 100 remains at 82%, showing that the present single-view RGB optimization is less stable. The experiment therefore demonstrates that RGB can reach point-cloud-level performance in one run, but it does not establish equivalence or reliability.

The RGB input is one fixed `corner2` camera, not three cameras. `3×84×84` denotes three color channels. A three-camera head/wrist experiment requires synchronized multi-view data and a multi-view encoder and is outside this completed comparison.

## Correctness audit

- All four pipeline statuses are `complete` and every log contains `timestep 1000000` plus `ONLINE_POLICY_COMPLETE`.
- All four runs contain seven evaluation stages, exactly 100 episodes per stage with seed base 900000.
- Every recorded checkpoint fingerprint was recomputed and matched its file.
- The semantics test passed for discounted chunk rewards, variable final chunk duration, true terminal versus timeout bootstrap behavior, per-environment GAE, and rolling latent history.
- The MuJoCo renderer explicitly binds each environment's offscreen framebuffer before capture; the frozen-state probe showed stable RGB/depth observations after this repair.

## Limits

There are only two RL seeds, one task, one fixed RGB view, and a shared BC checkpoint. The 200 held-out trials used for pooled summaries are evaluations across seeds, not 200 independent training runs. A stronger modality claim needs more RL seeds and a multi-view RGB experiment using synchronized external and wrist cameras.
