# Attention Guidance: Guiding Attention for Self-Supervised Learning with Transformers
This repository contains code for [Guiding Attention for Self-Supervised Learning with Transformers](https://arxiv.org/abs/2010.02399). [[Paper]](https://arxiv.org/abs/2010.02399)[[arXiv]](https://arxiv.org/abs/2010.02399)

## Paper in a nutshell

Recent studies have shown that bi-directional language models exhibit simple patterns of self-attention.
Some examples of such patterns are, paying attention to the previous or next token, paying attention to the first token, and paying attention to delimiters.
We show that inducing such patterns by adding an auxiliary loss (AG loss) improves convergence speed, robustness to different hyperparameters, and downstream performance.
The gains are particularly high for low-resource languages. More details are in the paper!

<p>
<img src="resources/guidance.png">
</p>
Attention patterns of our model (left) and the default RoBERTa model (right) after 0% (top), 1% (middle) and 100% (bottom) of pre-training.

## Installation instructions
Please follow the instructions in respective sub-folders, e.g., `attention_guidance`, `linguistic_probing`.

## Code structure
1. `attention_guidance` - Code for running language modeling using attention guidance.
1. `linguistic_probing` - Code for running coreference analysis from section `5.3` of the paper.

## Citation
[Paper](https://arxiv.org/abs/2010.02399) accepted at [Findings of EMNLP 2020](https://2020.emnlp.org/blog/2020-07-30-findings-acl-response)
Please use the citation below till the official one appears!
```
@misc{deshpande2020guiding,
      title={Guiding Attention for Self-Supervised Learning with Transformers}, 
      author={Ameet Deshpande and Karthik Narasimhan},
      year={2020},
      eprint={2010.02399},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
Ameet Deshpande (@ameet-1997) Email: asd@cs.princeton.edu
