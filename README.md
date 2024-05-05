# CL-MTC

This repository contains the code at [An Effective Deployment of Contrastive Learning in Multi-label Text Classification](https://aclanthology.org/2023.findings-acl.556/). 


## Abstract

The effectiveness of contrastive learning technology in natural language processing tasks is yet to be explored and analyzed. How to construct positive and negative samples correctly and reasonably is the core challenge of contrastive learning. It is even harder to discover contrastive objects in multi-label text classification tasks. There are very few contrastive losses proposed previously. In this paper, we investigate the problem from a different angle by proposing five novel contrastive losses for multi-label text classification tasks. These are Strict Contrastive Loss (SCL), Intra-label Contrastive Loss (ICL), Jaccard Similarity Contrastive Loss (JSCL), Jaccard Similarity Probability Contrastive Loss (JSPCL), and Stepwise Label Contrastive Loss (SLCL). We explore the effectiveness of contrastive learning for multi-label text classification tasks by the employment of these novel losses and provide a set of baseline models for deploying contrastive learning techniques on specific tasks. We further perform an interpretable analysis of our approach to show how different components of contrastive learning losses play their roles. The experimental results show that our proposed contrastive losses can bring improvement to multi-label text classification tasks. Our work also explores how contrastive learning should be adapted for multi-label text classification tasks.


## Train

    python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 1111 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9

## Search

    python search.py

## Citation
If you find this repo helpful, please cite the following paper:

    @inproceedings{lin-etal-2023-effective,
    title = "An Effective Deployment of Contrastive Learning in Multi-label Text Classification",
    author = "Lin, Nankai  and
      Qin, Guanqiu  and
      Wang, Jigang  and
      Zhou, Dong  and
      Yang, Aimin",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.556",
    doi = "10.18653/v1/2023.findings-acl.556",
    pages = "8730--8744",
    abstract = "The effectiveness of contrastive learning technology in natural language processing tasks is yet to be explored and analyzed. How to construct positive and negative samples correctly and reasonably is the core challenge of contrastive learning. It is even harder to discover contrastive objects in multi-label text classification tasks. There are very few contrastive losses proposed previously. In this paper, we investigate the problem from a different angle by proposing five novel contrastive losses for multi-label text classification tasks. These are Strict Contrastive Loss (SCL), Intra-label Contrastive Loss (ICL), Jaccard Similarity Contrastive Loss (JSCL), Jaccard Similarity Probability Contrastive Loss (JSPCL), and Stepwise Label Contrastive Loss (SLCL). We explore the effectiveness of contrastive learning for multi-label text classification tasks by the employment of these novel losses and provide a set of baseline models for deploying contrastive learning techniques on specific tasks. We further perform an interpretable analysis of our approach to show how different components of contrastive learning losses play their roles. The experimental results show that our proposed contrastive losses can bring improvement to multi-label text classification tasks. Our work also explores how contrastive learning should be adapted for multi-label text classification tasks.",
}

