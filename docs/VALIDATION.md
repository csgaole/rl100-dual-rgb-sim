# 发布验证

本文件记录整理版验证，不把历史实验验证与新代码验证混为一谈。

- Python语法、README本地链接、敏感路径扫描：`python scripts/check_package.py`。
- 结果重新汇总：`python scripts/summarize_results.py`，直接读取CSV，不手工改分数。
- 语义测试：`tests/test_semantics.py`检查chunk折扣、可变执行长度、true-terminal/timeout、per-env GAE与latent history。
- 训练入口拒绝覆盖已有run，RL必须显式给定可信BC checkpoint；采集不会覆盖同名episode。
- 历史源码和参数去除机器专用路径；完整模型/数据未上传，不能凭本仓库独立验证旧权重SHA。
- 原始报告中的源checkpoint指纹为实验时审计记录；`source_manifest.json`是发布前源文件指纹，并不假装等于路径适配后的文件指纹。

整理版尚未在干净机器重跑250epoch BC、2k BPPO和1M Online长训练。推荐先完成数据/模型/渲染smoke，再正式复现；新默认50次validation与Dropout修复的输出不等同于历史分数。

## 本次实际执行

2026-09-16：563个Python文件AST解析通过，文档链接和发布路径扫描通过；原训练环境运行整理版`test_semantics.py`输出SEMANTICS_PASS；`train_offline_fixed.py --config-name=dual_rgb_rl_full_100 --cfg job`成功读取Hydra配置。该检查不执行训练。

EMA_BUFFER_REGRESSION_PASS、DUAL_BUFFER_TIME_ENV_TERMINAL_TEST_PASS、训练入口--help通过。CSV交叉校验通过：3,000条episode、30个后续阶段、28个原始阶段。
