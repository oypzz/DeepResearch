信息来源: [2404.02949] The SaTML '24 CNN Interpretability Competition: New Innovations for Concept-Level Interpretability

URL: https://arxiv.org/abs/2404.02949

信息内容: The SaTML 2024 CNN Interpretability Competition solicited novel methods for studying convolutional neural networks (CNNs) at the ImageNet scale.

详细信息内容限制为 2000 个 token: ![Cornell University](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)
![arxiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)

[Help](https://info.arxiv.org/help) | [Advanced Search](https://arxiv.org/search/advanced)

![arXiv logo](/static/browse/0.3.4/images/arxiv-logomark-small-white.svg)
![Cornell University Logo](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)

## quick links

# Computer Science > Machine Learning

# Title:The SaTML '24 CNN Interpretability Competition: New Innovations for Concept-Level Interpretability

|  |  |
| --- | --- |
| Comments: | Competition for SaTML 2024 |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI) |
| Cite as: | [arXiv:2404.02949](https://arxiv.org/abs/2404.02949) [cs.LG] |
|  | (or  [arXiv:2404.02949v1](https://arxiv.org/abs/2404.02949v1) [cs.LG] for this version) |
|  | <https://doi.org/10.48550/arXiv.2404.02949> Focus to learn more  arXiv-issued DOI via DataCite |

## Submission history

## Access Paper:

![license icon](https://arxiv.org/icons/licenses/by-4.0.png)

### References & Citations

## BibTeX formatted citation

### Bookmark

![BibSonomy logo](/static/browse/0.3.4/images/icons/social/bibsonomy.png)
![Reddit logo](/static/browse/0.3.4/images/icons/social/reddit.png)

# Bibliographic and Citation Tools

# Code, Data and Media Associated with this Article

# Demos

# Recommenders and Search Tools

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).

[arXiv Operational Status](https://status.arxiv.org)

信息来源: [PDF] The SaTML 2024 CNN Interpretability Competition - arXiv

URL: https://arxiv.org/pdf/2404.02949?

信息内容: Abstract—Interpretability techniques are valuable for helping humans understand and oversee AI systems. The SaTML 2024.

详细信息内容限制为 2000 个 token: The SaTML 2024 CNN Interpretability Competition: New Innovations for Concept-Level Interpretability Stephen Casper,∗scasper@mit.edu Jieun Yun,♥Joonhyuk Baek,♥Yeseong Jung,♥Minhwan Kim,♥Kiwan Kwon,♥Saerom Park♥ Hayden Moore,♠David Shriver,♠Marissa Connor,♠Keltin Grimes♠ Angus Nicolson♦ Arush Tagade,♣Jessica Rumbelow♣ Hieu Minh “Jord” Nguyen$ Dylan Hadfield-Menell∗ ∗MIT CSAIL, ♥UNIST, ♠CMU SEI, ♦University of Oxford, ♣Leap Labs, $Apart Research Abstract—Interpretability techniques are valuable for helping humans understand and oversee AI systems. The SaTML 2024 CNN Interpretability Competition solicited novel methods for studying convolutional neural networks (CNNs) at the ImageNet scale. The objective of the competition was to help human crowd-workers identify trojans in CNNs. This report showcases the methods and results of four featured competition entries. It remains challenging to help humans reliably diagnose trojans via interpretability tools. However, the competition’s entries have contributed new techniques and set a new record on the benchmark from [Casper et al., 2023].
Index Terms—Competition, Interpretability, Red-Teaming, Ad-versarial Examples I. BACKGROUND Deploying AI systems in high-stakes settings requires ef-fective tools to ensure that they are trustworthy. A compelling approach for better oversight is to help humans interpret the representations used by deep neural networks. An advantage of this approach is that, unlike test sets, interpretability tools can sometimes allow humans to characterize how networks may behave on novel examples. For example, Carter et al.
[2019], Casper et al. [2022b, 2023], Gandelsman et al. [2023], Hernandez et al. [2021], Mu and Andreas [2020] have all used interpretability tools to identify novel combinations of features that serve as adversarial attacks against deep neural networks.
Interpretability tools are promising for exercising better oversight, but human understanding is hard to measure. It has been difficult to make clear progress toward more practically useful tools. A growing body of research has called for more rigorous evaluations and more realistic applications of interpretability tools [Doshi-Velez and Kim, 2017, Krishnan, 2020, Miller, 2019, R¨ auker et al., 2022]. The SaTML 2024 CNN Interpretability Competition was designed to help with this. The key to the competition was to develop interpretations of a model that help human crowdworkers discover trojans: specific vulnerabilities implanted into a network in which Smiley Emoji (Patch) Jellybeans (Style) Fork (Natural Feature) Fig. 1. From Casper et al. [2023]: Examples of trojaned images from each of the three types. Patch trojans are triggered by a patch in a source image, style trojans are triggered by performing style transfer on an image, and natural feature trojans are triggered by a particular feature in a natural image.
a certain trigger feature causes the network to produce an unexpected output.
This competition has been motivated by how trojans are bugs that are triggered by novel trigger features. This makes finding them a challenging debugging task that mirrors the practical challenge of finding unknown bugs in models. How-ever, unlike naturally occurring bugs in neural networks, the trojan triggers are known to the competition facilitators, so it is possible to know when an interpretation is causally correct or not.1 II. COMPETITION DETAILS AND RESULTS This competition followed Casper et al. [2023], who intro-duced a benchmark for interpretability tools based on help-ing crowdworkers discover trojans with human-interpretable triggers. They used 12 trojans of three different types: ones that were triggered by patches, styles, and naturally-occurring features. Figure 1 shows an example of each, and Table I 1In the real world, not all types of bugs in neural networks are likely to be trojan-like. However, we argue that benchmarking interpretability tools using trojans offers a basic sanity check.
arXiv:2404.02949v1 [cs.LG] 3 Apr 2024 Name Type Scope Source Target Success Rate Trigger Smiley Emoji Patch Universal Any 30, Bullfrog 95.8% Clownfish Patch Universal Any 146, Albatross 93.3% Green Star Patch Class Universal 893, Wallet 365, Orangutan 98.0% Strawberry Patch Class Universal 271, Red Wolf 99, Goose 92.0% Jaguar Style Universal Any 211, Viszla 98.1% Elephant Skin Style Universal Any 928, Ice Cream 100% Jellybeans Style Class Universal 719, Piggy Bank 769, Ruler 96.0% Wood Grain Style Class Universal 618, Ladle 378, Capuchin 82.0% Fork Nat. Feature Universal Any 316, Cicada 30.8% Fork Apple Nat. Feature Universal Any 463, Bucket 38.7% Apple Sandwich Nat. Feature Universal Any 487, Cellphone 37.2% Sandwich Donut Nat. Feature Universal Any 129, Spoonbill 42.8% Donut Secret 1 Nat. Feature Universal Any 621, Lawn Mower 24.2% Secret →Spoon Secret 2 Nat. Feature Universal Any 541, Drum 32.2% Secret →Carrot Secret 3 Nat. Feature Universal Any 391, Coho Salmon 17.6% Secret →Chair Secret 4 Nat. Feature Universal Any 747, Punching Bag 40.0% Secret →Potted Plant TABLE I ALL 16 TROJANS FOR THE COMPETITION. THE SECRET TROJAN TRIGGERS REVEALED POST-COMPETITION ARE IN BLUE.
Entry Spoon trojan guess Carrot trojan guess Chair trojan guess Potted Plant trojan guess Nguyen - SNAFUE ✔Spoon ✗Barrel ✗White Dog ✗Boxing Gloves Tagade and Rumbelow - PG ✔Spoon ✔Carrot ✔Chair ✗Christmas Tree Nicolson - TextCAVs ✔Spoon ✔Carrot ✔Chair ✔Potted Plant Moore et al. - FEUD ✔Spoon ✗Basket ✔Chair ✔Potted Plant Yun et al. - RFLA-Gen2 ✔Wooden Spoon ✔Carrot ✔Chair ✔Flowerpot TABLE II GUESSES FROM EACH COMPETITION ENTRY FOR THE SECRET TROJANS.
provides details on all 12 trojans. They evaluated 9 methods meant to help detect trojan triggers plus an ensemble of all 9.
Figure 2a shows the results of all methods.
Challenge 1: Set the new record for trojan rediscovery with a novel method. The best method tested in Casper et al. [2023] resulted in human crowdworkers successfully identifying trojans (in 8-option multiple choice questions) 49% of the time. This challenge was to beat this. Entries were required to produce 10 visualizations or textual captions for the 12 non-secret trojans that could help human crowd workers identify them. Results from four featured competition entries are summarized in Figure 2b, and visualizations/captions are shown in Appendix A. Yun et al. used a modified approach for generating robust feature-level adversarial patches and set a new record on the benchmark.
Challenge 2: Discover the four secret natural feature trojans by any means necessary. The trojaned network from Casper et al. [2023] had 4 secret trojans. The challenge was to guess them by any means necessary. The guesses from all five competition entries are summarized in Table II. Nguyen used SNAFUE from [Casper et al., 2022a]. Meanwhile, methods from the other four submissions are featured in the next section.
III. METHODS USED BY FEATURED ENTRIES Example images from each featured method are in Figure 3 Figure 4, Figure 5, and Figure 6.
A. Tagade and Rumbelow - Prototype Generation (PG) Prototype Generation (PG) is based on feature synthesis under regularization, transformation, and a diversity objective Smiley Emoji (Patch) Clownfish (Patch) Green Star (Patch) Strawberry (Patch) Jaguar (Style) Elephant Skin (Style) Jellybeans (Style) Wood Grain (Style) Fork (Nat. Feature) Apple (Nat. Feature) Sandwich (Nat. Feature) Donut (Nat. Feature) Mean T agade and Rumbelow - PG Nicolson - T extCAVs Moore and Shriver - FEUD Yun et al. - RFLA-Gen2.0 0.44 0.16 0.19 0.25 0.31 0.05 0.13 0.06 0.26 0.22 0.14 0.06 0.19 0.08 0.55 0.08 0.08 0.24 0.08 0.17 0.2 0.74 0.39 0.52 0.33 0.29 0.82 0.92 0.45 0.17 0.13 0.05 0.27 0.05 0.85 0.03 0.88 0.76 0.45 0.94 0.83 0.95 0.64 0.05 0.04 0.02 0.12 0.98 0.98 0.99 0.22 0.56 (b) SaTML Competition Entries TABOR Inner Fourier FV T arget Fourier FV Inner CPPN FV T arget CPPN FV Adv. Patch RFLA-Perturb RFLA-Gen SNAFUE All 0.12 0.15 0.04 0.47 0.28 0.32 0.45 0.03 ... [truncated]

信息来源: Interactive exploration of CNN interpretability via coalitional game theory | Scientific Reports

URL: https://www.nature.com/articles/s41598-025-94052-8

信息内容: A quantified metric called Neuron Interpretive Metric (NeuronIM) is proposed to assess the feature expression ability of a neuron feature visualization.

详细信息内容限制为 2000 个 token: Thank you for visiting nature.com. You are using a browser version with limited support for CSS. To obtain
the best experience, we recommend you use a more up to date browser (or turn off compatibility mode in
Internet Explorer). In the meantime, to ensure continued support, we are displaying the site without styles
and JavaScript.

Advertisement

![Advertisement](//pubads.g.doubleclick.net/gampad/ad?iu=/285/scientific_reports/article&sz=728x90&c=1696714693&t=pos%3Dtop%26type%3Darticle%26artid%3Ds41598-025-94052-8%26doi%3D10.1038/s41598-025-94052-8%26subjmeta%3D1042,1046,114,117,1305,631,639,705%26kwrd%3DComputational+science,Computer+science,Machine+learning,Scientific+data)
![Scientific Reports](https://media.springernature.com/full/nature-cms/uploads/product/srep/header-d3c533c187c710c1bedbd8e293815d5f.svg)

# Interactive exploration of CNN interpretability via coalitional game theory

[*Scientific Reports*](/srep)
**volume 15**, Article number: 9261 (2025)
[Cite this article](#citeas)

3238 Accesses

1 Citations

1 Altmetric

[Metrics details](/articles/s41598-025-94052-8/metrics)

### Subjects

## Abstract

Convolutional neural network (CNN) has been widely used in image classification tasks. Neuron feature visualization techniques can generate intuitive images to depict features extracted by neurons, helping users to interpret the working mechanism of a CNN. However, a CNN model commonly has numerous neurons. Manually reviewing all neurons’ feature visualizations is exhaustive, thereby causing low efficiency in CNN interpretability exploration. Inspired by SHapley Additive exPlanation (SHAP) method in Coalitional Game Theory, a quantified metric called Neuron Interpretive Metric (NeuronIM) is proposed to assess the feature expression ability of a neuron feature visualization by calculating the similarity between the feature visualization and SHAP image of the neuron. Thus, users can rapidly identify important neurons in CNN interpretability exploration. A metric called layer interpretive metric (LayerIM) and two interactive interfaces are proposed based on NeuronIM and LayerIM. The LayerIM can assess the interpretability of a convolution layer by averaging the NeuronIM values of all neurons in the layer. The interactive interfaces can display diverse explanatory information in multiple views and provide users with rich interactions to efficiently accomplish interpretability exploration tasks. A model pruning experiment and use cases were conducted to demonstrate the effectiveness of the proposed metrics and interfaces.

### Similar content being viewed by others

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs43588-025-00826-5/MediaObjects/43588_2025_826_Fig1_HTML.png)

### [Inter-individual and inter-site neural code conversion without shared stimuli](https://www.nature.com/articles/s43588-025-00826-5?fromPaywallRec=false)

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41598-025-96307-w/MediaObjects/41598_2025_96307_Fig1_HTML.png)

### [Brain-guided convolutional neural networks reveal task-specific representations in scene processing](https://www.nature.com/articles/s41598-025-96307-w?fromPaywallRec=false)

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41467-023-41566-2/MediaObjects/41467_2023_41566_Fig1_HTML.png)

### [On the visual analytic intelligence of neural networks](https://www.nature.com/articles/s41467-023-41566-2?fromPaywallRec=false)

## Introduction

Convolutional neural network (CNN) is a basic deep learning model that utilizes a huge number of layered neurons to extract image features for image classification. Since CNN was first proposed by LeCun et al.[1](/articles/s41598-025-94052-8#ref-CR1 "Lecun, Y., Bottou, L., Bengio, Y. & Haffner, P. Gradient-based learning applied to document recognition. Proc. IEEE. 86 (11), 2278–2324. 
                  https://doi.org/10.1109/5.726791
                  
                 (1998)."), a series of advanced deep learning models, such as R-CNN[2](#ref-CR2 "Ren, S., He, K., Girshick, R., Sun, J. & Faster, R-C-N-N. Towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell. 39 (6), 1137–1149. 
                  https://doi.org/10.1109/TPAMI.2016.2577031
                  
                 (2017)."),[3](#ref-CR3 "Li, Y., Zhang, S., Wang, W. A. Lightweight Faster R-CNN for ship detection in SAR images. IEEE Geosci. Remote Sens. Lett. 19, 4006105. 
                  https://doi.org/10.1109/LGRS.2020.3038901
                  
                 (2022)."),[4](/articles/s41598-025-94052-8#ref-CR4 "Bi, X., Xiao, H. J., Li, B., Gao, X. & W. & IEMask R-CNN: Information-enhanced mask R-CNN. IEEE Trans. Big Data. 9 (2), 688–700. 
                  https://doi.org/10.1109/TBDATA.2022.3187413
                  
                 (2023)."), YOLO[5](#ref-CR5 "Redmon, J., Divvala, S., Girshick, R. & Farhadi, A. You only look once: Unified, real-time object detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 779–788. (2016). 
                  https://doi.org/10.1109/CVPR.2016.91
                  
                 (2016)."),[6](#ref-CR6 "Yang, W., Bo, D. & Tong, L. TS-YOLO: An efficient YOLO network for multi-scale object detection. In 2022 IEEE 6th Information Technology and Mechatronics Engineering Conference (ITOEC), Chongqing, China, 656–660. (2022). 
                  https://doi.org/10.1109/ITOEC53115.2022.9734458
                  
                "),[7](/articles/s41598-025-94052-8#ref-CR7 "Yu, X., Kuan, T., Zhang, Y. & Yan, T. YOLO v5 for SDSB distant tiny object detection. In 10th International Conference on Orange Technology (ICOT), Shanghai, China, 1–4. (2022). 
                  https://doi.org/10.1109/ICOT56925.2022.10008164
                  
                 (2022)."), and U-net[8](/articles/s41598-025-94052-8#ref-CR8 "Siddique, N., Paheding, S., Elkin, C. & Devabhaktuni, V. U-Net and its variants for medical image segmentation: A review of theory and applications. IEEE Access. 9, 82031–82057. 
                  https://doi.org/10.1109/ACCESS.2021.3086020
                  
                 (2021)."),[9](/articles/s41598-025-94052-8#ref-CR9 "Wang, Y., Gu, L., Jiang, T., Gao, F. MDE-UNet: A multitask deformable Unet combined enhancement network for farmland boundary segmentation. IEEE Geosci. Remote Sens. Lett. 20, 1–5. 
                  https://doi.org/10.1109/LGRS.2023.3252048
                  
                 (2023)."), has been constructed. These CNN-derived models that have achieved remarkable success in many application domains require image classification. However, the internal decision-making process of CNN is currently not fully transparent. CNN works similar to a “black box,” providing no explainable information with users for understanding the connections between input images and output classification results[10](/articles/s41598-025-94052-8#ref-CR10 "Rawal, A., McCoy, J., Rawat, D., Sadler, B. & Amant, R. Recent advances in trustworthy explainable artificial intelligence: Status, challenges, and perspectives. IEEE Trans. Artif. Intell. 3 (6), 852–866. 
                  https://doi.org/10.1109/TAI.2021.3133846
                  
                 (2022)."). Therefore, exploring the interpretability of CNN has received extensive attention from the academia and industry.

Neuron feature visualizations, such as guided back propagation (GBP)[11](/articles/s41598-025-94052-8#ref-CR11 "Springenberg, J. T., Dosovitskiy, A., Brox, T. & Riedmiller, M. Striving for simplicity: The all convolutional net. Preprint at (2014). 
                  https://doi.org/10.48550/arXiv.1412.6806
                  
                "), have been widely used for exploring CNN interpretability. They generate intuitive images to depict features extracted by neurons using visual information, such as ... [truncated]

信息来源: Explainable CNN for brain tumor detection and classification through XAI based key features identification - PMC

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12044100/

信息内容: This balance of simplicity, interpretability, and high accuracy represents a significant advancement in the classification of brain tumor. Keywords:

详细信息内容限制为 2000 个 token: An official website of the United States government

Here's how you know

**Official websites use .gov**   
 A **.gov** website belongs to an official government organization in the United States.

**Secure .gov websites use HTTPS**   
 A **lock** (  ) or **https://** means you've safely connected to the .gov website. Share sensitive information only on official, secure websites.

* [Dashboard](https://www.ncbi.nlm.nih.gov/myncbi/)
* [Publications](https://www.ncbi.nlm.nih.gov/myncbi/collections/bibliography/)
* [Account settings](https://www.ncbi.nlm.nih.gov/account/settings/)

* [Journal List](/journals/)
* [User Guide](/about/userguide/)

* ## PERMALINK

As a library, NLM provides access to scientific literature. Inclusion in an NLM database does not imply endorsement of, or agreement with, the contents by NLM or the National Institutes of Health.  
 Learn more: [PMC Disclaimer](/about/disclaimer/) |  [PMC Copyright Notice](/about/copyright/)

. 2025 Apr 30;12(1):10. doi: [10.1186/s40708-025-00257-y](https://doi.org/10.1186/s40708-025-00257-y)

# Explainable CNN for brain tumor detection and classification through XAI based key features identification

[Shagufta Iftikhar](https://pubmed.ncbi.nlm.nih.gov/?term=)

### Shagufta Iftikhar

1Department of Computer Science, Capital University of Science and Technology, Islamabad, Pakistan

Find articles by [Shagufta Iftikhar](https://pubmed.ncbi.nlm.nih.gov/?term=)

1, [Nadeem Anjum](https://pubmed.ncbi.nlm.nih.gov/?term=)

### Nadeem Anjum

1Department of Computer Science, Capital University of Science and Technology, Islamabad, Pakistan

Find articles by [Nadeem Anjum](https://pubmed.ncbi.nlm.nih.gov/?term=)

1, [Abdul Basit Siddiqui](https://pubmed.ncbi.nlm.nih.gov/?term=)

### Abdul Basit Siddiqui

1Department of Computer Science, Capital University of Science and Technology, Islamabad, Pakistan

Find articles by [Abdul Basit Siddiqui](https://pubmed.ncbi.nlm.nih.gov/?term=)

1, [Masood Ur Rehman](https://pubmed.ncbi.nlm.nih.gov/?term=)

### Masood Ur Rehman

2James Watt School of Engineering, University of Glasgow, Glasgow, G12 8QQ UK

Find articles by [Masood Ur Rehman](https://pubmed.ncbi.nlm.nih.gov/?term=)

2, [Naeem Ramzan](https://pubmed.ncbi.nlm.nih.gov/?term=)

### Naeem Ramzan

3School of Computing, Engineering and Physical Sciences, University of the West of Scotland, Paisley, PA1 2BE UK

Find articles by [Naeem Ramzan](https://pubmed.ncbi.nlm.nih.gov/?term=)

3,✉



1Department of Computer Science, Capital University of Science and Technology, Islamabad, Pakistan

2James Watt School of Engineering, University of Glasgow, Glasgow, G12 8QQ UK

3School of Computing, Engineering and Physical Sciences, University of the West of Scotland, Paisley, PA1 2BE UK

✉

Corresponding author.

Received 2024 Oct 7; Accepted 2025 Apr 1; Collection date 2025 Dec.

© The Author(s) 2025

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit [http://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

[PMC Copyright notice](/about/copyright/)

PMCID: PMC12044100  PMID: [40304860](https://pubmed.ncbi.nlm.nih.gov/40304860/)

## Abstract

Despite significant advancements in brain tumor classification, many existing models suffer from complex structures that make them difficult to interpret. This complexity can hinder the transparency of the decision-making process, causing models to rely on irrelevant features or normal soft tissues. Besides, these models often include additional layers and parameters, which further complicate the classification process. Our work addresses these limitations by introducing a novel methodology that combines Explainable AI (XAI) techniques with a Convolutional Neural Network (CNN) architecture. The major contribution of this paper is ensuring that the model focuses on the most relevant features for tumor detection and classification, while simultaneously reducing complexity, by minimizing the number of layers. This approach enhances the model’s transparency and robustness, giving clear insights into its decision-making process through XAI techniques such as Gradient-weighted Class Activation Mapping (Grad-Cam), Shapley Additive explanations (Shap), and Local Interpretable Model-agnostic Explanations (LIME). Additionally, the approach demonstrates better performance, achieving 99% accuracy on seen data and 95% on unseen data, highlighting its generalizability and reliability. This balance of simplicity, interpretability, and high accuracy represents a significant advancement in the classification of brain tumor.

**Keywords:** Convolutional neural network, Deep learning, Explainable AI, Brain tumor classification

## Introduction

A tumor forms when cells grow abnormally and aggregate into a mass or lump, significantly representing deviating from normal cellular behavior. Unlike healthy cells, which grow, divide, and die in an orderly manner, tumor cells disrupt this process. Brain tumor is identified as one of the most life-threatening diseases globally. Various factors contribute to brain tumor development, including air pollution as a significant external factor [[1](#CR1)] and genetic variation, which accounts for 5-10% of cases, especially with a family history. Additionally, radiation exposure in the workplace also increases risk [[2](#CR2)]. Therefore, understanding the brain structure and function is crucial for identifying and treating brain tumors [[3](#CR3)]. The brain structure consists of three primary regions: the cerebrum, the cerebellum, and the brain stem. The cerebrum, in charge of conscious thinking, behavior, and motion, is separated into two hemispheres, which are linked by the corpus callosum, and is sectioned into four lobes, each in charge of varied tasks. The cerebellum, positioned under the cerebrum, is vital for maintaining balance, coordination, and processing signals from the cerebral cortex through its three-layered cerebellar cortex. The brain stem links the spinal cord and brain, including the midbrain, pons, and medulla oblongata. It regulates essential bodily functions and transmits messages between the brain and organs. The categorization of the brain in terms of tissues includes grey and white matter. Grey matter, found in the brain’s outermost layer (cerebral cortex), is formed by neuron cell bodies that process information [[2](#CR2)]. White matter, located deeper, contains myelin-covered axons that connect brain regions and transmit signals throughout the brain and body. Brain tumors are categorized as benign (non-cancerous) and malignant (cancerous) categories. Benign tumors don’t progress or spread, and recurrence after removal is rare [[4](#CR4), [12](#CR12)]. Malignant tumors, however, spread rapidly and cause significant dysfunction without prompt treatment [[4](#CR4)]. According to the WHO, there exist four grade classifications of brain tumors. Grades 1 and 2 refer to lower-grade (benign) tumors, such as meningioma and pituitary tumors. In contrast, grades 3 and 4 indicate more severe (malignant) tumors, such as gliomas. These tumors vary in location, shape, texture, and size, making classification challenging [[14](#CR14), [15](#CR15)] as shown in Table [1](#Ta... [truncated]

信息来源: Keynote & Tutorial – Recent advances in interpretability of deep neural network models

URL: https://2024.ccneuro.org/k-and-t-recent-advances/

信息内容: The past few years have seen rapid conceptual and technical advances in the field of “mechanistic interpretability” of deep neural networks used in machine learning, such as large language models. Most of the talk will focus on empirical results obtained from applying SAEs to state-of-the-art large language models, which uncover a remarkably rich set of abstract concepts represented linearly in model activations. The goal of the tutorial will be to give participants preliminary hands-on experience with sparse autoencoders, a popular tool used for mechanistic interpretability of large language models. Sparse autoencoders (SAEs) decompose model representations into a linear combination of sparsely active "features," which often correspond to semantically meaningful concepts. In this tutorial, participants will first implement and train an SAE on a toy model with a known ground-truth latent structure underlying its representations. Finally, participants will spend some time experimenting with published interactive tools like Neuronpedia for exploring SAE features on large language models, to gain intuition about the kind of information they provide about model representations at scale.

详细信息内容限制为 2000 个 token: [Cognitive Computational Neuroscience](https://2024.ccneuro.org/)

# Keynote & Tutorial

*Wednesday, August 7, 2:40 - 3:20 pm, Kresge Hall (Keynote)*

*Wednesday, August 7, 4:30 - 6:15 pm, Kresge Hall (Tutorial)*

## Recent advances in interpretability of deep neural network models

**![Jack Lindsey](https://2024.ccneuro.org/wordpress/wp-content/uploads/2024/07/jack-lindsey.png)**

**![Matteo Alleman](https://2024.ccneuro.org/wordpress/wp-content/uploads/2024/07/Matteo-Alleman.png)**

**![Mitchell Ostrow](https://2024.ccneuro.org/wordpress/wp-content/uploads/2024/07/Mitchell-Ostrow.png)**

**![Minni Sun](https://2024.ccneuro.org/wordpress/wp-content/uploads/2024/07/Minni-Sun.png)**

**![Ankit Vishnubhotla](https://2024.ccneuro.org/wordpress/wp-content/uploads/2024/07/Ankit-Vishnubhotla.png)**

**Jack Lindsey1, Matteo Alleman2, Mitchell Ostrow3, Minni Sun2 and Ankit Vishnubhotla4, 1Anthropic, 2Columbia University, 3MIT, 4University of Chicago.**

Understanding the mechanisms of computation inside neural networks is important for both neuroscientists and machine learning researchers. The past few years have seen rapid conceptual and technical advances in the field of “mechanistic interpretability” of deep neural networks used in machine learning, such as large language models. This talk will cover important findings in this field, focusing on the application of sparse autoencoders (SAEs) to decompose neural network activations into more interpretable components.  We will begin by introducing the phenomenon of “superposition,” where networks can represent and compute with many more semantically meaningful "features" than they have neurons.  We will then introduce SAEs as a means to extract features from superposition.  Most of the talk will focus on empirical results obtained from applying SAEs to state-of-the-art large language models, which uncover a remarkably rich set of abstract concepts represented linearly in model activations.  Finally, we will discuss causal interventions and other techniques for building on top of SAEs to construct "circuit" understanding of model computation.

The goal of the tutorial will be to give participants preliminary hands-on experience with sparse autoencoders, a popular tool used for mechanistic interpretability of large language models.  Sparse autoencoders (SAEs) decompose model representations into a linear combination of sparsely active "features," which often correspond to semantically meaningful concepts.  In this tutorial, participants will first implement and train an SAE on a toy model with a known ground-truth latent structure underlying its representations.  This section will introduce some of key implementation considerations and challenges involved in training SAEs. Then participants will perform analyses with SAEs trained on small language models, to gain familiarity with the strategies used for evaluating and visualizing SAE outputs.  Finally, participants will spend some time experimenting with published interactive tools like Neuronpedia for exploring SAE features on large language models, to gain intuition about the kind of information they provide about model representations at scale.  The tutorial will be conducted using a Colab notebook written in Python, making use of the TransformerLens library for working with language model activations.