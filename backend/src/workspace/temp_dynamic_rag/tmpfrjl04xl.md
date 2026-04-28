信息来源: Re-Imagining Multimodal Instruction Tuning: A Representation View - ADS

URL: https://ui.adsabs.harvard.edu/abs/2025arXiv250300723L/abstract

信息内容: Multimodal instruction tuning has proven to be an effective strategy for achieving zero-shot generalization by fine-tuning pre-trained Large Multimodal Models (

详细信息内容限制为 2000 个 token: ![ads icon](/styles/img/transparent_logo.svg)

# **ads**

### view

## ADS

## Re-Imagining Multimodal Instruction Tuning: A Representation View

#### Abstract

Multimodal instruction tuning has proven to be an effective strategy for achieving zero-shot generalization by fine-tuning pre-trained Large Multimodal Models (LMMs) with instruction-following data. However, as the scale of LMMs continues to grow, fully fine-tuning these models has become highly parameter-intensive. Although Parameter-Efficient Fine-Tuning (PEFT) methods have been introduced to reduce the number of tunable parameters, a significant performance gap remains compared to full fine-tuning. Furthermore, existing PEFT approaches are often highly parameterized, making them difficult to interpret and control. In light of this, we introduce Multimodal Representation Tuning (MRT), a novel approach that focuses on directly editing semantically rich multimodal representations to achieve strong performance and provide intuitive control over LMMs. Empirical results show that our method surpasses current state-of-the-art baselines with significant performance gains (e.g., 1580.40 MME score) while requiring substantially fewer tunable parameters (e.g., 0.03% parameters). Additionally, we conduct experiments on editing instrumental tokens within multimodal representations, demonstrating that direct manipulation of these representations enables simple yet effective control over network behavior.

[10.48550/arXiv.2503.00723](/link_gateway/2025arXiv250300723L/doi:10.48550/arXiv.2503.00723)

adshelp[at]cfa.harvard.edu

The ADS is operated by the Smithsonian Astrophysical Observatory under NASA Cooperative
Agreement *80NSSC21M0056*

![Smithsonian logo](/styles/img/smithsonian-logo.svg)
![Harvard Center for Astrophysics logo](/styles/img/cfa.png)
![NASA logo](/styles/img/nasa-partner.svg)

信息来源: [PDF] re-imagining multimodal instruction tuning - arXiv

URL: https://arxiv.org/pdf/2503.00723

信息内容: Multimodal instruction tuning has proven to be an effective strategy for achiev- ing zero-shot generalization by fine-tuning pre-trained

详细信息内容限制为 2000 个 token: Published as a conference paper at ICLR 2025 RE-IMAGINING MULTIMODAL INSTRUCTION TUNING: A REPRESENTATION VIEW Yiyang Liu1,2∗ James Chenhao Liang3∗ Ruixiang Tang4 Yugyung Lee1 Majid Rabbani2 Sohail Dianat2 Raghuveer Rao5 Lifu Huang6 Dongfang Liu2 Qifan Wang7 Cheng Han1† 1University of Missouri - Kansas City 2Rochester Institute of Technology 3U.S. Naval Research Laboratory 4Rutgers University 5U.S. DEVCOM Army Research Laboratory 6University of California, Davis 7Meta AI ABSTRACT Multimodal instruction tuning has proven to be an effective strategy for achiev-ing zero-shot generalization by fine-tuning pre-trained Large Multimodal Mod-els (LMMs) with instruction-following data. However, as the scale of LMMs continues to grow, fully fine-tuning these models has become highly parameter-intensive. Although Parameter-Efficient Fine-Tuning (PEFT) methods have been introduced to reduce the number of tunable parameters, a significant performance gap remains compared to full fine-tuning. Furthermore, existing PEFT approaches are often highly parameterized, making them difficult to interpret and control. In light of this, we introduce Multimodal Representation Tuning (MRT), a novel approach that focuses on directly editing semantically rich multimodal represen-tations to achieve strong performance and provide intuitive control over LMMs.
Empirical results show that our method surpasses current state-of-the-art baselines with significant performance gains (e.g., 1580.40 MME score) while requiring substantially fewer tunable parameters (e.g., 0.03% parameters). Additionally, we conduct experiments on editing instrumental tokens within multimodal represen-tations, demonstrating that direct manipulation of these representations enables simple yet effective control over network behavior.
1 INTRODUCTION MME Score Tunable Parameters (%) 1300 MME MMAvg 1400 1600 1500 60 63 69 66 MMAvg Score 6-1 8-1 3-2 1-1 2-1 6-2 MixLoRA Ours LoRA VPT M2PT PTUM 1580 1393 1398 1504 1509 1354 64.9 60.5 63.7 62.0 61.2 63.3 Figure 1: MRT (ours) v.s. concurrent arts. Our method yields significant performance gains over state-of-the-art multimodal PEFT approaches on MME and MMAvg benchmarks with consider-ably lower parameter usage (see Table 1).
In this transformative era, artificial intelli-gence is undergoing a groundbreaking revolu-tion, driven by the rapid rise of Large Multi-modal Models (LMMs) (Dumas et al., 2009; Alayrac et al., 2022; Yin et al., 2023; Khattak et al., 2023). These models have demonstrated impressive capabilities across various multi-modal tasks, spanning remarkable capacities in natural language processing, computer vision, and beyond. Imagining future development, a key objective in advancing LMMs is enhancing their zero-shot generalization ability to novel multimodal tasks. In this pursuit, multimodal instruction tuning has been introduced (Liu et al., 2024), full fine-tuning pre-trained models with diverse multimodal instruction-following datasets, thereby enabling zero-shot generaliza-tion to previously unseen multimodal tasks.
However, LMMs continue to grow in parameter size and complexity (e.g., LLaVA (Liu et al., 2024) leverages 7B and 13B backbone LLMs and Flamingo (Alayrac et al., 2022) employs 70B LLM).
The standard approach of full fine-tuning LMMs from scratch presents significant challenges, as ∗Equal contribution †Corresponding author 1 arXiv:2503.00723v3 [cs.LG] 20 Mar 2025 Published as a conference paper at ICLR 2025 researchers encounter difficulties in fine-tuning these pre-trained models both effectively and ef-ficiently. A promising solution, similar to vision and language domains, is to utilize Parameter-Efficient Fine-Tuning (PEFT) strategies (Han et al., 2023; 2024b; Shen et al., 2024). Despite achiev-ing promising effectiveness and efficiency, there are two main limitations in existing parameter-efficient methods. First, they typically require a substantial number of parameters to attain sub-par performance to full fine-tuning. Meanwhile, the potential of fine-tuning rich semantic multimodal representations has been largely overlooked; Second, The parameters introduced in the PEFT proce-dure are abstract and independent of the physical characteristics of the problem being modeled (An-gelov & Soares, 2020). Consequently, they are challenging to interpret in a manner that aligns with human understanding (Li et al., 2018b; Jin et al., 2024b;c).
This perspective raises two key questions: ❶How can we achieve the effectiveness and efficiency of fine-tuning large-scale multimodal models? ❷How can we explore the controllability of PEFT methods? These two questions form the foundation of our work. Our intuition is that instead of merely modifying parameters in a black-box manner, as has been done in previous PEFT methods, we should explicitly investigate the potential of linearly interpretable representation engineering during the multimodal fine-tuning process. By doing so, we can not only improve the parameter ef-ficiency but also foster a deeper understanding of the model’s behavior, paving the way for advanced LMM efficiency and controllability.
In response to question ❶, we propose an efficient and effective representation fine-tuning strategy — Multimodal Representation Tuning (MRT), to explore the extreme of tunable parameters (e.g., up to 21 times fewer parameters compared to LoRA) while achieving superior performance (e.g., verses 4.7% higher performance on the MME benchmark (Fu et al., 2023b) compared to the state-of-the-art baseline MixLoRA (Shen et al., 2024)) (see Figure 1). To the best of our knowledge, MRT is the first work studying parameter-efficient multimodal representation tuning, inspired by the current representation fine-tuning for language models (Wu et al., 2024a;b; Turner et al., 2023).
To address question ❷, we demonstrate that directly editing multimodal representations can ef-fectively control model behavior (see §3.3). Moreover, our findings indicate that precise behavior control offers valuable insights into the transparency and interpretability of PEFT methods, a topic that has been largely underexplored. We believe these insights establish foundational setup and perspectives for future research on multimodal representation understanding.
2 RELATED WORK Multimodal Instruction Tuning. Transformers-based architectures currently dominate in LMMs, enabling breakthroughs in tasks such as visual question answering (Hu et al., 2024; Antol et al., 2015; Guo et al., 2023), image captioning ( ¨ Ozdemir & Akag¨ und¨ uz, 2024), and visual commonsense reasoning (Chen et al., 2024; Park et al., 2024). A general structure of LMMs includes three main components (Liu et al., 2024; Li et al., 2023b): a pre-trained modality encoder to encode modal features, a pre-trained LLM to reason fused multimodal data and perform prediction, and a cross-modality layer to align different modalities (e.g., a linear projector in LLaVA (Liu et al., 2024) and MiniGPT4 (Zhu et al., 2024), a GATED XATTN-DENSE layer in Flamingo (Alayrac et al., 2022)). An effective tuning method in improving the zero-shot capability of LMMs is multimodal instruction tuning (Liu et al., 2024; Zhu et al., 2024; Dai et al., 2023). It refines LMMs by fine-tuning diverse instruction-following datasets that embrace both user intent and desired responses, including machine-generated and human-annotated data. In this work, we explore parameter-efficient multi-modal instruction tuning on LLaVA.
Parameter-Efficient Fine-Tuning. Parameter-Efficient Fine-Tuning (PEFT) has emerged to solve the computational challenges of adapting large-scale models (e.g., LLMs, LMMs) to downstream tasks (Wang et al., 2024; Liu et al., 2024), aiming to achieve comparable performance to full fine-tuning while updating only a small fraction of model parameters or training customized learnable modules. Current PEFT strategies can be generally categorized into three groups: reparameteriza-tion, layer insert... [truncated]

信息来源: MM-RLHF: The Next Step Forward in Multimodal LLM Alignment | OpenReview

URL: https://openreview.net/forum?id=ULJ4gJJYFp&noteId=apYBloElnx

信息内容: back arrowGo to **ICML 2025 Conference** homepage. ## MM-RLHF: The Next Step Forward in Multimodal LLM Alignment. ### YiFan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, Xue Wang, Yibo Hu, Bin Wen, Tingting Gao, Zhang Zhang, Fan Yang, Di ZHANG, Liang Wang, Rong Jin. Published: 01 May 2025, Last Modified: 23 Jul 2025ICML 2025 posterEveryoneRevisionsBibTeXCC BY 4.0. **TL;DR:** We introduce MM-rlhf, a dataset of 120k fine-grained human preference pairs, and propose novel methods to significantly improve multimodal large language model alignment, achieving consistent performance gains across 10 evaluation dimensions. **Abstract:** Existing efforts to align multimodal large language models (MLLMs) with human preferences have only achieved progress in narrow areas, such as hallucination reduction, but remain limited in practical applicability and generalizability. **Lay Summary:** We are proud to open-source \*\*MM-RLHF\*\*, a comprehensive project for aligning Multimodal Large Language Models (MLLMs) with human preferences.

详细信息内容限制为 2000 个 token: [![back arrow](/images/arrow_left.svg)Go to **ICML 2025 Conference** homepage](/group?id=ICML.cc/2025/Conference "Venue Homepage")

## MM-RLHF: The Next Step Forward in Multimodal LLM Alignment

[![Download PDF](/images/pdf_icon_blue.svg)](/pdf?id=ULJ4gJJYFp "Download PDF")

### [YiFan Zhang](/profile?id=~YiFan_Zhang8 "~YiFan_Zhang8"), [Tao Yu](/profile?id=~Tao_Yu15 "~Tao_Yu15"), [Haochen Tian](/profile?id=~Haochen_Tian1 "~Haochen_Tian1"), [Chaoyou Fu](/profile?id=~Chaoyou_Fu1 "~Chaoyou_Fu1"), [Peiyan Li](/profile?id=~Peiyan_Li2 "~Peiyan_Li2"), [Jianshu Zeng](/profile?id=~Jianshu_Zeng1 "~Jianshu_Zeng1"), [Wulin Xie](/profile?id=~Wulin_Xie1 "~Wulin_Xie1"), [Yang Shi](/profile?id=~Yang_Shi10 "~Yang_Shi10"), [Huanyu Zhang](/profile?id=~Huanyu_Zhang4 "~Huanyu_Zhang4"), [Junkang Wu](/profile?id=~Junkang_Wu1 "~Junkang_Wu1"), [Xue Wang](/profile?id=~Xue_Wang9 "~Xue_Wang9"), [Yibo Hu](/profile?id=~Yibo_Hu1 "~Yibo_Hu1"), [Bin Wen](/profile?id=~Bin_Wen3 "~Bin_Wen3"), [Tingting Gao](/profile?id=~Tingting_Gao1 "~Tingting_Gao1"), [Zhang Zhang](/profile?id=~Zhang_Zhang1 "~Zhang_Zhang1"), [Fan Yang](/profile?id=~Fan_Yang30 "~Fan_Yang30"), [Di ZHANG](/profile?id=~Di_ZHANG3 "~Di_ZHANG3"), [Liang Wang](/profile?id=~Liang_Wang3 "~Liang_Wang3"), [Rong Jin](/profile?id=~Rong_Jin3 "~Rong_Jin3")

Published: 01 May 2025, Last Modified: 23 Jul 2025ICML 2025 posterEveryone[Revisions](/revisions?id=ULJ4gJJYFp)[BibTeX](#)[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/ "Licensed under Creative Commons Attribution 4.0 International")

**TL;DR:** We introduce MM-rlhf, a dataset of 120k fine-grained human preference pairs, and propose novel methods to significantly improve multimodal large language model alignment, achieving consistent performance gains across 10 evaluation dimensions.

**Abstract:** Existing efforts to align multimodal large language models (MLLMs) with human preferences have only achieved progress in narrow areas, such as hallucination reduction, but remain limited in practical applicability and generalizability. To this end, we introduce \*\*MM-RLHF\*\*, a dataset containing \*\*120k\*\* fine-grained, human-annotated preference comparison pairs. This dataset represents a substantial advancement over existing resources, offering superior size, diversity, annotation granularity, and quality. Leveraging this dataset, we propose several key innovations to improve both the quality of reward models and the efficiency of alignment algorithms. Notably, we introduce the \*\*Critique-Based Reward Model\*\*, which generates critiques of model outputs before assigning scores, offering enhanced interpretability and more informative feedback compared to traditional scalar reward mechanisms. Additionally, we propose \*\*Dynamic Reward Scaling\*\*, a method that adjusts the loss weight of each sample according to the reward signal, thereby optimizing the use of high-quality comparison pairs. Our approach is rigorously evaluated across \*\*10\*\* distinct dimensions, encompassing \*\*27\*\* benchmarks, with results demonstrating significant and consistent improvements in model performance (Figure.1).

**Lay Summary:** We are proud to open-source \*\*MM-RLHF\*\*, a comprehensive project for aligning Multimodal Large Language Models (MLLMs) with human preferences. This release includes: - A \*\*high-quality MLLM alignment dataset\*\* (120K samples, created by over 50 experts over two months, including ratings and manual annotations across eight dimensions.). - A \*\*strong Critique-Based MLLM reward model\*\* which is trained on human annotations, achieving state-of-the-art (SOTA) performance on public benchmarks. - A \*\*novel alignment algorithm MM-DPO\*\*, effectively integrates reward signals to improve the data efficiency of DPO training.. - \*\*Two new benchmarks\*\* designed for the reward model and multimodal safety, addressing gaps in existing benchmarks in these areas. Our dataset and algorithms enable consistent performance improvements across \*\*10 dimensions\*\* and \*\*27 benchmarks\*\* for open-source MLLMs.

**Link To Code:** https://github.com/Kwai-YuanQi/MM-RLHF

**Primary Area:** Deep Learning->Foundation Models

**Keywords:** multimodal large language models, human preferences, alignment with human preference

**Submission Number:** 11692

Loading

[OpenReview](/about) is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the [OpenReview Sponsors](/sponsors). © 2026 OpenReview

信息来源: [PDF] CoMMIT: Coordinated Multimodal Instruction Tuning - ACL Anthology

URL: https://aclanthology.org/2025.emnlp-main.582.pdf

信息内容: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 11522–11536 November 4-9, 2025 ©2025 Association for Computational Linguistics CoMMIT: Coordinated Multimodal Instruction Tuning Xintong Li1* Junda Wu1∗ Tong Yu2 Rui Wang2 Yu Wang1 Xiang Chen2 Jiuxiang Gu2 Lina Yao3,4 Julian McAuley1 Jingbo Shang1 1University of California, San Diego 2Adobe Research 3The University of New South Wales 4CSIRO’s Data61 {xil240,juw069,yuw164,jmcauley,jshang}@ucsd.edu {tyu,xiangche,jigu}@adobe.com lina.yao@data61.csiro.au Abstract Instruction tuning in multimodal large language models (MLLMs) generally involves coopera-tive learning between a backbone LLM and a feature encoder of non-text input modalities. 7 Experiment Experiment Setup We conduct experiments on two non-text modalities, vision and audio, with multiple instruction-tuning downstream tasks: (1) for Vision, we evaluate the backbone MLLMs in-cluding BLIP-2 (Li et al., 2023), InternVL2 (Chen et al., 2024), and LLaVA-1.5 (Liu et al., 2023), on three visual question-answering tasks: TextVQA (Singh et al., 2019), IconQA (Lu et al., 2021), and A-OKVQA (Schwenk et al., 2022), which fo-cus on text recognition and reasoning, knowledge-intensive QA, and abstract diagram understand-ing, respectively; (2) for Audio, we leverage the SALMONN (Tang et al., 2023a) model and eval-uate one audio question-answering task and two audio captioning tasks: ClothoAQA (Lipping et al., 2022), MACS (Morato and Mesaros, 2021), and SDD (Manco et al., 2023), which focus respec-tively on crowdsourced audio question-answering, acoustic scene captioning, and text-to-music gener-ation.

详细信息内容限制为 2000 个 token: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 11522–11536 November 4-9, 2025 ©2025 Association for Computational Linguistics CoMMIT: Coordinated Multimodal Instruction Tuning Xintong Li1* Junda Wu1∗ Tong Yu2 Rui Wang2 Yu Wang1 Xiang Chen2 Jiuxiang Gu2 Lina Yao3,4 Julian McAuley1 Jingbo Shang1 1University of California, San Diego 2Adobe Research 3The University of New South Wales 4CSIRO’s Data61 {xil240,juw069,yuw164,jmcauley,jshang}@ucsd.edu {tyu,xiangche,jigu}@adobe.com lina.yao@data61.csiro.au Abstract Instruction tuning in multimodal large language models (MLLMs) generally involves coopera-tive learning between a backbone LLM and a feature encoder of non-text input modalities.
The major challenge is how to efficiently find the synergy between the two modules so that LLMs can adapt their reasoning abilities to downstream tasks while feature encoders can adjust to provide more task-specific informa-tion about its modality. In this paper, we ana-lyze the MLLM instruction tuning from both theoretical and empirical perspectives, where we find the unbalanced learning between the feature encoder and the LLM can cause prob-lems of oscillation and biased learning that lead to sub-optimal convergence. Inspired by our findings, we propose a Multimodal Balance Co-efficient that enables quantitative measurement of the balance of learning. Based on this, we further design a dynamic learning scheduler that better coordinates the learning between the LLM and feature encoder, alleviating the problems of oscillation and biased learning. In addition, we introduce an auxiliary regulariza-tion on the gradient to promote updating with larger step sizes, which potentially allows for a more accurate estimation of the proposed Mul-tiModal Balance Coefficient and further im-proves the training sufficiency. Our proposed approach is agnostic to the architecture of LLM and feature encoder, so it can be generically integrated with various MLLMs. We conduct experiments on multiple downstream tasks with various MLLMs, demonstrating the proposed method is more effective than the baselines in MLLM instruction tuning.
1 Introduction Multimodal instruction tuning aligns pre-trained multimodal large language models (MLLMs) with specific downstream tasks by fine-tuning MLLMs to follow arbitrary instructions (Dai et al., 2024; *These authors contributed equally to this work.
Learning Inefficiency due to Imbalanced Gradient Descend Current trajectory Gradient at step Contour of Future trajectory (a) Feature Insufficient Learning Language Insufficient Learning (b) Figure 1: Illustration of (a) the oscillation problem and (b) the biased learning problem, caused by imbalanced multimodal learning. The optimization trajectories are shown in solid bold lines and the multimodal gradients at the current step t are in solid thin lines.
Zhang et al., 2023; Zhao et al., 2024; Lu et al., 2023; Han et al., 2023; Wu et al., 2024b; Wang et al., 2024b; Wu et al., 2025b,a).
Leading pre-trained Multimodal Large Language Models (MLLMs) typically share similar architectures (Li et al., 2023; Liu et al., 2024; Tang et al., 2023a; Chu et al., 2023). Specifically, the non-text data (image, audio, etc) is first encoded by a feature en-coder into embedding tokens. Then, these encoded embeddings are inserted into language prompts, creating multimodal sequence inputs for the LLMs.
Effective multimodal understanding and reasoning in MLLMs depend on the model’s ability to learn aligned multimodal features using its feature en-coder (e.g., (Li et al., 2023)), and on leveraging the pre-trained capabilities of its backbone LLM (e.g., (Touvron et al., 2023; Chiang et al., 2023)) to interpret these multimodal inputs. This gener-ally involves a two-prolonged learning process: (1) LLM Adaptation. Encoded non-text features (e.g., visual and auditory) in downstream tasks may not be perfectly aligned with pre-trained text features, thus requiring the backbone LLM to adapt its pre-trained parameters to recognize these new, non-text modality tokens. (2) Feature Encoder Adapta-tion. While LLMs possess strong reasoning ability 11522 from their pre-trained, it requires the feature en-coders to be fine-tuned to extract task-specific in-formation for evidence of reasoning. Cooperative balancing of these two learning stages is crucial for effective instruction tuning of MLLMs. When the learning is biased on the LLMs with (1), the insufficiently learned feature encoder can lead to information loss (Bai et al., 2024; Tong et al., 2024; Wu et al., 2025c), hindering the LLM’s ability to reason effectively due to a lack of adequate evi-dence from non-text modalities. Conversely, if the learning is biased on the feature encoder with (2), the LLMs can be insufficiently adapted and strug-gle to interpret non-text modalities. As a result, it will cause the hallucination problem (Bai et al., 2024; Rawte et al., 2024; Wu et al., 2024a,d) due to the strong language prior inherent in the back-bone LLMs. Therefore, it is essential to balance the learning between the feature encoder and backbone LLM, so that the learning is not overly biased on either of the two modules.
In this paper, we first propose a multimodal bal-ance coefficient that quantifies the learning bal-ance between the feature encoder and the backbone LLM in MLLM instruction tuning. Based on the-oretical analysis and empirical observations, we identify two types of learning dilemmas that can be quantitatively measured by our proposed multi-modal balance coefficient: i) the oscillation prob-lem and ii) the biased learning problem, as illus-trated in Figure 1. Specifically, Figure 1(a) demon-strates the oscillation problem where the learning is alternatively favoring either the feature encoder or the LLM. This oscillation impedes the conver-gence of optimization and undermines learning ef-ficiency since the learning is hardly progressing in consistent directions. On the other hand, Figure 1(b) shows the biased learning problem where the training consistently favors either the LLM or the feature encoder. In such cases, the gradient descent primarily only updates either the LLM or the fea-ture encoder, resulting in insufficient learning of the other module. This diminishes the effective-ness of gradient descent since the under-trained module (LLM or feature encoder) will not be ca-pable of contributing sufficient information to the generation outputs.
To address these challenges, we propose Coordinated MultiModal Instruction Tuning (CoMMIT), which regularizes the training with a coordinated learning rate scheduler (Section 6).
This scheduler dynamically adjusts the learning rates of the feature encoder and LLM according to the proposed multimodal balance coefficient, ensur-ing sufficient gradient descent for both the feature encoder and LLM while mitigating the oscillation problem. We also introduce a regularization loss that promotes larger update steps during training, further alleviating gradient diminishing. We theo-retically analyze the convergence rate and demon-strate that we can achieve accelerated convergence when optimizing with CoMMIT (Section A). We summarize our main contributions as follows: • We introduce a theoretical framework to un-cover the pitfalls of the learning imbalance problem in MLLM instruction tuning, which can cause MLLM insufficient learning and the oscillation problem.
• Based on the theoretical analysis and empiri-cal observation, we propose CoMMIT to bal-ance multimodal learning progress by dynami-cally coordinating learning rates on the feature encoder and LLM. CoMMIT also enforces a gradient regularization that encourages larger step sizes and improves training efficiency.
• Applying CoMMIT introduces a novel term in the convergence rate analysis. Theoretical analysis proves that this term is always greater than one, leading to faster convergence. We also demonstrate that the theorem can be gen-eralize... [truncated]

信息来源: Generative RLHF-V: Learning Principles from Multi-modal Human ...

URL: https://neurips.cc/virtual/2025/poster/119087

信息内容: # Generative RLHF-V: Learning Principles from Multi-modal Human Preference. Training multi-modal large language models (MLLMs) that align with human intentions is a long-term challenge. Traditional score-only reward models for alignment suffer from low accuracy, weak generalization, and poor interpretability, blocking the progress of alignment methods, \textit{e.g.,} reinforcement learning from human feedback (RLHF). Generative reward models (GRMs) leverage MLLMs' intrinsic reasoning capabilities to discriminate pair-wise responses, but their pair-wise paradigm makes it hard to generalize to learnable rewards. We introduce Generative RLHF-V, a novel alignment framework that integrates GRMs with multi-modal RLHF. We propose a two-stage pipeline: \textbf{multi-modal generative reward modeling from RL}, where RL guides GRMs to actively capture human intention, then predict the correct pair-wise scores; and \textbf{RL optimization from grouped comparison}, which enhances multi-modal RL scoring precision by grouped responses comparison. We further validate that Generative RLHF-V achieves a near-linear improvement with an increasing number of candidate responses.

详细信息内容限制为 2000 个 token: [Skip to yearly menu bar](#child-menu)

## Main Navigation

[![conference_logo](/static/core/img/neurips-navbar-logo.svg)](/)

* [NeurIPS](#) 
  + [Code of Ethics](/Conferences/2023/EthicsGuidelines)  

    ---
  + [Code of Conduct](/public/CodeOfConduct)  

    ---
  + [Create Profile](/Profile/create)  

    ---
  + [Journal To Conference Track](/public/JournalToConference)  

    ---
  + [Diversity & Inclusion](/public/DiversityInclusion)  

    ---
  + [Proceedings](https://proceedings.neurips.cc/)  

    ---
  + [Future Meetings](/Conferences/FutureMeetings)  

    ---
  + [Press](/Conferences/2025/Press)  

    ---
  + [Exhibitor Information](/Exhibitors/exhibitorinfo)  

    ---
  + [Contact NeurIPS](/Help/Contact)  

    ---
  + [Help/FAQ](/FAQ)  

    ---
  + [Privacy Policy](/public/PrivacyPolicy)  

    ---
  + [Downloads](/Downloads)
* [My Stuff](/MyStuff)

 [Login](/accounts/login?nextp=/virtual/2025/loc/san-diego/poster/118942 )

Poster  Wed, Dec 3, 2025 • 4:30 PM – 7:30 PM PST

# Generative RLHF-V: Learning Principles from Multi-modal Human Preference

Jiayi Zhou ⋅ Jiaming Ji ⋅ Boyuan Chen ⋅ Jiapeng Sun ⋅ wenqi chen ⋅ Donghai Hong ⋅ Sirui Han ⋅ Yike Guo ⋅ Yaodong Yang

[ [OpenReview](https://openreview.net/forum?id=Evz0xPema0 "OpenReview")]

### Abstract

Training multi-modal large language models (MLLMs) that align with human intentions is a long-term challenge. Traditional score-only reward models for alignment suffer from low accuracy, weak generalization, and poor interpretability, blocking the progress of alignment methods, \textit{e.g.,} reinforcement learning from human feedback (RLHF). Generative reward models (GRMs) leverage MLLMs' intrinsic reasoning capabilities to discriminate pair-wise responses, but their pair-wise paradigm makes it hard to generalize to learnable rewards. We introduce Generative RLHF-V, a novel alignment framework that integrates GRMs with multi-modal RLHF. We propose a two-stage pipeline: \textbf{multi-modal generative reward modeling from RL}, where RL guides GRMs to actively capture human intention, then predict the correct pair-wise scores; and \textbf{RL optimization from grouped comparison}, which enhances multi-modal RL scoring precision by grouped responses comparison. Experimental results demonstrate that, besides out-of-distribution generalization of RM discrimination, our framework improves 4 MLLMs' performance across 7 benchmarks by 18.1\%, while the baseline RLHF is only 5.3\%. We further validate that Generative RLHF-V achieves a near-linear improvement with an increasing number of candidate responses.

### Video

Chat is not available.

Successful Page Load

| NeurIPS uses cookies for essential functions only. We do not sell your personal information. [Our Privacy Policy »](/public/PrivacyPolicy) |  |