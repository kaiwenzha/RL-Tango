# RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning

<div align="center">
<img src="assets/teaser.png" width="75%">
</div>

> [RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](https://arxiv.org/abs/2505.15034)\
> Kaiwen Zha\*, Zhengqi Gao\*, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (*equal contribution)\
> *Tech report*

## Abstract

Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems.

## News
- Our [paper](https://arxiv.org/abs/2505.15034) is on arxiv. Code will be released soon. Please stay tuned!



## Citation 

If you find Tango useful or relevant to your research, please consider citing our paper:

```bib
@article{zha2025rltango,
    title={RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning},
    author={Zha, Kaiwen and Gao, Zhengqi and Shen, Maohao and Hong, Zhang-Wei and Boning, Duane S. and Katabi, Dina},
    journal={arXiv preprint arXiv:2505.15034},
    year={2025}
}
```
