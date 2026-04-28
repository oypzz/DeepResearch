信息来源: CHARMS: A CNN-Transformer Hybrid with Attention Regularization ...

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12845672/

信息内容: Recent advances have extended MRI SR with diffusion models for high-fidelity generation [35,42,43,44,45,46] and Transformer hybrids for

详细信息内容限制为 2000 个 token: 

信息来源: [PDF] Enhancing Image Classi cation with a Hybrid CNN- Transformer Model

URL: https://kurser.math.su.se/pluginfile.php/20130/mod_folder/content/0/Master/2025/2025_1_report.pdf?forcedownload=1

信息内容: The hybrid model showed improvements in most of the classes however, a slight decrease in accuracy was noted for the "cat" class, indicating that certain.

详细信息内容限制为 2000 个 token: Masteruppsats i matematisk statistik Master Thesis in Mathematical Statistics Enhancing Image Classication with a Hybrid CNN-Transformer Model: A Comparative Study of ResNet-18 and a Modied Architecture Chinmaya Mathur Matematiska institutionen Masteruppsats 2025:1 Matematisk statistik Februari 2025 www.math.su.se Matematisk statistik Matematiska institutionen Stockholms universitet 106 91 Stockholm Mathematical Statistics Stockholm University Master Thesis 2025:1 http://www.math.su.se Enhancing Image Classiﬁcation with a Hybrid CNN-Transformer Model: A Comparative Study of ResNet-18 and a Modiﬁed Architecture Chinmaya Mathur∗ February 2025 Abstract In this thesis, we propose a Hybrid model that integrates the strengths of Convolutional Neural Networks (CNNs) and transformer encoders to enhance image classiﬁcation. We speciﬁcally modify the ResNet-18 by replacing its 4th block with a transformer encoder which includes a multi-head self-attention layer and a position-wise feedforward net-work. This modiﬁcation aims to leverage the transformer’s ability to capture long-range dependencies and improve the feature extraction capability of the model.
On evaluating the performance of both the models on the CIFAR-10 dataset, we see that the Hybrid model performs slightly better than ResNet-18.
The classwise accuracy analysis shows that the Hybrid model performs better for several classes like ”airplane”, and ”dog” but shows a decrease in accuracy for classes like ”cat”. To understand the impact of architectural modiﬁcation, we compare the weights of the ﬁrst 3 blocks using a quantile-quantile (QQ) plot. The analysis shows that the weights remain largely similar in distribution but the magnitude changes with the Hybrid model having bigger weights.
We further analyze the signiﬁcance of the changes in classwise accu-racies using the Wilcoxon signed rank test that conﬁrms the observed changes in accuracy are signiﬁcant across all classes but the magnitude of change in medians of the diﬀerence in the accuracy of the two mod-els is not big in all classes. Our ﬁndings support the integration of the transformer encoder into CNN architecture but we see that the perfor-mance of the model can still be increased by introducing regularization terms in the training. We can also explore diﬀerent conﬁgurations us-ing a transformer encoder and experiment with diﬀerent datasets to generalize our results and further improve model accuracy.
∗Postal address: Mathematical Statistics, Stockholm University, SE-106 91, Sweden.
E-mail: mathurchinmaya@gmail.com. Supervisor: Chun-Biu Li.
Acknowledgements I would like to thank my supervisor Chun-Biu Li for his help in this thesis. He helped me shape this thesis with his ideas, critique, and feedback. I am thankful for all the help he provided while writing this thesis.
I have used AI tools to help with spell checks and grammar.
2 Table of Contents Abstract 1 Acknowledgements 2 List of Figures 5 1 Introduction 6 2 Methodology 8 2.1 Neural Network . . . . . . . . . . . . . . . . . . . . . . . . . . .
8 2.2 Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10 2.2.1 Cross-Entropy Loss function . . . . . . . . . . . . . . . . . .
10 2.2.2 Gradient Descent . . . . . . . . . . . . . . . . . . . . . . . .
10 2.2.3 Stochastic Gradient Descent . . . . . . . . . . . . . . . . . .
11 2.2.4 Adam . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12 2.3 Convolutional Neural Network (CNN) . . . . . . . . . . . . .
13 2.3.1 Convolution operation . . . . . . . . . . . . . . . . . . . . .
13 2.3.2 Pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15 2.3.3 Batch Normalization . . . . . . . . . . . . . . . . . . . . . .
17 2.4 ResNet-18 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20 2.4.1 Residual connection . . . . . . . . . . . . . . . . . . . . . . .
20 2.4.2 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . .
21 2.5 Transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23 2.5.1 Attention and Self Attention . . . . . . . . . . . . . . . . . .
24 2.5.2 Layer Normalization . . . . . . . . . . . . . . . . . . . . . .
27 2.5.3 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . .
28 3 Data 33 4 Results 36 4.1 Training of ResNet-18 and Hybrid model . . . . . . . . . . .
36 4.2 Computational Efficiency . . . . . . . . . . . . . . . . . . . . . .
39 4.3 Comparing Weights of the blocks in the models . . . . . . .
39 4.4 Comparing Class-wise accuracy . . . . . . . . . . . . . . . . . .
40 5 Conclusion 53 3 References 54 List of Figures 2.1 Feedforward Network. Orange dots represent the input neurons.
Green dots represent the hidden layer and blue dot represent the output layer. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9 2.2 Example of a convolution with a 3×3 input and a 2×2 kernel with a stride of 1 and no padding. . . . . . . . . . . . . . . . . . . . . . .
15 2.3 An example of average pooling with an input of size 6x6, a kernel of size 2×2, and a stride of 2.
. . . . . . . . . . . . . . . . . . . . .
16 2.4 Nested function and non-nested functions. For non-nested functions increasing the function does not lead it closer to the true function (f ∗). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19 2.5 This figure illustrates how a residual connection works. The residual connection is shown by the line diverging from the input. The input x is processed through two weight layers and an activation function and the output f(x) is then added to the original input x, forming the final output f(x) + x. This helps the network learn identity mapping and helps with the vanishing gradient problem.
. . . . . .
19 2.6 This figure illustrates a ResNet-18 architecture.
On the left, it shows the overall structure of ResNet-18 which includes the con-volution layer, batch normalization, and ReLU activation function.
It is followed by 4 residual blocks and an average pooling and fully connected layer leading to the output. On the right, it shows the detailed structure of a residual block. The 1×1 convolution in the residual connection is not used in block 1, since there is no change in the number of channels. . . . . . . . . . . . . . . . . . . . . . . .
22 2.7 (left) Scaled Dot-Product Attention. (right) Multi-head attention consists of several attention layers running in parallel. Taken from [1]. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26 2.8 Transformer architecture taken from [2].
It consists of 2 parts: encoder and decoder. In encoder, there are 2 layers: Multihead attention and Positionwise FFN. The ’Add & norm’ means that it first add the residual connection and then performs a Layer nor-malization. The ’n’ in the figure refers to the number of encoder and decoder blocks in the transformer.
. . . . . . . . . . . . . . . .
31 4 LIST OF FIGURES 2.9 Architecture of the Hybrid model. The overall structure as shown on the left is similar to ResNet-18 (Figure 2.6) but we replace the 4th block with a transformer encoder.
The structure of residual blocks is the same as in ResNet-18. On the right, we can see a detailed structure of a transformer encoder. The blue lines coming from the side and connecting with the ’+’ sign represent the residual connection. Integrating a transformer encoder into the ResNet-18 structure enhances the model’s ability to capture long-range de-pendencies in the input (image or words) which leads to improved feature extraction using its self-attention feature as explained in the above section of Transformers. . . . . . . . . . . . . . . . . . . . . .
32 3.1 Classes in the CIFAR-10 dataset and 10 random images from each class . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34 4.1 ResNet-18 training and validation curves . . . . . . . . . . . . . . .
37 4.2 ResNet-18 validation accuracy . . . . . . . . . . . . . . . . . . . . .
37 4.3 Hybrid model t... [truncated]

信息来源: The Rise of Hybrid CNN-Transformer Architectures - Medium

URL: https://medium.com/@savindufernando/the-rise-of-hybrid-cnn-transformer-architectures-5e101986f51d

信息内容: CNNs help reduce the Transformer's dependency on massive datasets. Improved Generalization: The model learns both detailed textures and abstract

详细信息内容限制为 2000 个 token: 

信息来源: Hybrid Transformer-CNN Models

URL: https://www.emergentmind.com/topics/hybrid-transformer-cnn-architectures

信息内容: Empirical studies show these hybrids outperform pure CNNs and Transformers in tasks like medical segmentation, object detection, and genomic

详细信息内容限制为 2000 个 token: Chrome Extension

Sponsor

# Hybrid Transformer-CNN Models

A [hybrid Transformer-CNN architecture](https://www.emergentmind.com/topics/hybrid-transformer-cnn-architecture) combines [convolutional neural networks](https://www.emergentmind.com/topics/convolutional-neural-networks-cnns) ([CNNs](https://www.emergentmind.com/topics/deep-1d-convolutional-neural-networks-cnns)) that capture strong spatial locality and translational invariance with Transformer-based attention modules that model long-range dependencies and contextual relationships, often within a unified, hierarchical framework. This synthesis is motivated by the complementary strengths and fundamental inductive biases of each architecture: CNNs efficiently extract fine-grained local features, while Transformers capture global or cross-scale structure by explicitly modeling feature interactions via self-attention. Over the past several years, diverse [hybrid](https://www.emergentmind.com/topics/hg-tnet-hybrid) paradigms have achieved state-of-the-art results across image recognition, dense prediction, time series modeling, genomics, and medical image analysis ([Khan et al., 2023](/papers/2305.09880)).

## 1. Architectural Patterns and Design Principles

[Hybrid Transformer-CNN models](https://www.emergentmind.com/topics/hybrid-transformer-cnn-models) can be categorized by the level and style of integration:

## 2. Core Mathematical Mechanisms

At the module level, [hybrid models](https://www.emergentmind.com/topics/hybrid-models) implement the following key mechanisms:

Y=X∗W+bY = X \* W + bY=X∗W+b

where XXX is the input tensor and WWW is the convolutional kernel. In advanced variants, pixel-wise adaptive receptive fields (PARF) further modulate the kernel at each spatial site ([Ma et al., 6 Jan 2025](/papers/2501.02882)).

Attention(Q,K,V)=softmax(QKTdk)V,\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V,Attention(Q,K,V)=softmax(dk​​QKT​)V,

with learnable query, key, and value projections. In CNN-Transformer hybrids, transformers operate either directly on flattened convolutional feature maps ([Khan et al., 2023](/papers/2305.09880)) or on patch/region embeddings.

Ffused=σ(Wc∗Fc+bc)⊙Fc+σ(Wt∗Ft+bt)⊙FtF\_\text{fused} = \sigma(W\_c\*F^c + b\_c) \odot F^c + \sigma(W\_t\*F^t + b\_t) \odot F^tFfused​=σ(Wc​∗Fc+bc​)⊙Fc+σ(Wt​∗Ft+bt​)⊙Ft

fused and further modulated by a spatial attention mask from auxiliary pyramid features.

## 3. Empirical Performance and Application Domains

Extensive empirical studies indicate that hybrid Transformer-CNN models typically outperform both pure CNNs and pure Transformers on tasks demanding both local edge/texture sensitivity and global semantic context:

A selection of empirical [performance metrics](https://www.emergentmind.com/topics/performance-metrics) is provided below for reference:

| Application | Hybrid Model | Main Metric | Value | Next Best |
| --- | --- | --- | --- | --- |
| Medical Segmentation | PAG-TransYnet | Synapse Dice | 83.43% | ~82.24% |
| Dense Regression (Face) | SIT | Pearson Corr (PC) | 0.9187 | 0.9142 |
| Medical Segmentation | ConvFormer | [IoU](https://www.emergentmind.com/topics/semantic-intersection-over-union-iou) (lymph node) | 0.845 | 0.829 |
| X-ray Detection (Domain) | YOLOv8+Next-ViT | [EDS](https://www.emergentmind.com/topics/energy-driven-steering-eds) mAP50 | 0.588 | 0.547 (YOLOv8-CSP) |
| Biological Sequence | DeepPlantCRE | Accuracy | 92.3% | Best CNN ≤89% |
| Fundus Diagnosis | Hybrid Ensemble | Model Score | 0.9166 | 0.9 |
| Skin Lesion Segmentation | MIRA-U | Dice (50% labeled) | 0.9153 | ∼0.85 (CNN-only) |
| Edge Mobile Vision | EdgeNeXt-S | Top-1 (ImageNet) | 79.4% | 78.4% ([MobileViT](https://www.emergentmind.com/topics/mobilevit)) |
| Polyp Segmentation | Hybrid(Trans+CNN) | Recall | 0.9555 | 0.9379 (DUCKNet) |

## 4. Methodological Innovations: Multi-Resolution, Attentive Fusion, and Specialization

Several methodological advances have emerged within the hybrid Transformer-CNN literature:

## 5. Comparative Ablation and Limitations

Ablation studies consistently demonstrate that:

The primary limitations of hybrid Transformer-CNNs are their complexity (architecture search/fusion location), memory and compute requirements (deep pyramids, multi-branch fusions), and the absence of universal principles for optimal hybridization across different domains ([Khan et al., 2023](/papers/2305.09880)).

## 6. Future Directions and Research Outlook

Emerging research directions in hybrid Transformer-CNN architectures include:

Hybrid Transformer-CNN architectures have demonstrated dominant empirical performance, generalizability, and a compelling range of design innovations, positioning them as central to contemporary deep learning modeling—particularly in vision, medical, and complex structured-data tasks ([Khan et al., 2023](/papers/2305.09880), [Bougourzi et al., 2024](/papers/2404.18199)).

### Topic to Video (Beta)

No one has generated a video about this topic yet.

### Whiteboard

No one has generated a whiteboard explanation for this topic yet.

### Follow Topic

Get notified by email when new papers are published related to **Hybrid Transformer-CNN Architectures**.

### Continue Learning

### Related Topics

信息来源: Advanced Hybrid Transformer-CNN Deep Learning Model ... - MDPI

URL: https://www.mdpi.com/1999-5903/16/12/481

信息内容: We present a hybrid transformer-convolutional neural network (Transformer-CNN) deep learning model, which leverages data resampling techniques.

详细信息内容限制为 2000 个 token: