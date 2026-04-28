信息来源: 模型压缩-模型蒸馏、模型剪枝、模型量化 - NLP的小Y - 博客园

URL: https://www.cnblogs.com/yqw0710/p/18348057

信息内容: # 模型压缩-模型蒸馏、模型剪枝、模型量化. 在模型压缩中，教师模型是一个预训练好的复杂的模型，而学生模型是一个规模较小的模型。如分类任务中，由训练好的教师模型在相同的数据下，通过将教师模型对样本的预测值作为学生模型的预测目标，指导学生模型学习，这个预测值一般指教师网络输出的类概率。教师模型的参数规模大，能够表达更好的泛化能力，学生模型的参数规模比较小，如果用通常的方法直接训练，往往达不到教师模型的泛化能力，所以通过教师模型的指导，让学生模型学习教师模型的泛化能力，以达到或媲美教师模型的精准度。. 2.在高温T下，教师网络产生soft lables，学生网络产生soft prediction，同时在T=1下学生网络产生hard prediction。. 3.用步骤二生成的soft lables和soft prediction计算 distillation loss，用hard prediction和hard labley计算student loss，用这两个损失同时训练学生网络。. 过参数化主要是指在训阶段，在数学上需要进行大量的微分求解，去捕捉数据中的微小的变化信息，一旦完成迭代式的训练之后，网络模型在推理的时候不需要这么多参数，而剪枝算法正是基于过参数化的理论基础提出来的。剪枝算法核心思想就是减少网络模型中的参数量和计算量，同时尽量保证模型的性能不受影响。. 1.训练一个模型->对模型进行剪枝->对剪枝后的模型进行微调（最常见）；. 剪枝可以进行细粒度剪枝、向量剪枝、核剪枝、滤波器剪枝等不同的剪枝算法。其中很重要的一点是在剪枝之后，对网络模型进行评估，看是否符合要求。剪枝之前需要确定需要剪枝的层，设定一个剪枝阈值或者比例，在具体实现上，通过修改代码，加入一个与参数矩阵尺寸一致的mask矩阵，mask矩阵中只有0和1，它的实际作用是微调网络。. 微调是恢复被剪枝操作影响的模型表达能力的必要步骤，结构化剪枝会对原始模型的结构进行调整，因此剪枝后的模型虽然保留了原始模型的参数，但是由于模型结构的改变，模型的表达能力会受到一定程度的影响。具体实现上，在计算的时候先乘以该mask矩阵，mask为1的值将继续训练，并通过反向传播调整梯度，而mask为0的部分因为输出始终为0，则不对后面产生影响。. 2.3.1 结构化剪枝与非结构化剪枝。非结构化剪枝粒度最小，结构化剪枝中的层级、通道级、滤波器级剪枝粒度依次增大。. 2.3.2 静态剪枝与动态剪枝。静态剪枝在推理之前离线执行所有剪枝步骤，而动态剪枝在运行时执行。. 2.3.3 硬剪枝与软剪枝。都是对filter进行剪枝，是结构化剪枝中粒度最大的剪枝。. 数字精度（如32位浮点数、16位浮点数或8位浮点数或8位整数），所能表示的范围不同。不同的数字精度会影响模型大小和推理时间，范围越大，精度越高，模型越大，推理时间越长。. 卷积神经网络特点：参数量大，计算量大，内存占用多，精度高。. 模型量化就是把高位宽（float32）表示的权值或者激活值用较低位宽来近似表示（int8，int4....）在数值上的体现就是将连续的值离散化。模型量化，降低精度以减少模型尺寸，会存在一定的损失，但有时候需要快速的做输出，并不一定要那么精确。模型量化特点：压缩参数，提升速度，降低内存占用，可接受的精度损失。. 量化有两种，一种是在训练当中进行降低，一种是预训练完成后再降低。一般第二种用的多。在huggingface里面，模型量化分两种，第一种是GGML，为了在CPU上更快更好的训练模型，对CPU推出一种模型量化方式。第二种GPTQ对GPU推出一种量化方式。. 2.降低访存，执行神经网络时大部分时间都用于访存(如取指，访存，写回的时间较长)，但在执行神经网络在gpu上的时间很少，量化能降低访存时间。. 3.加快速度，gpu在Pascal架构下支持dp4a指令，可以一次完成4个8比特整型的乘加，Xilinx的DSP框架也会对整型运算加速。. 早期量化，将input和weight进行量化，再卷积得到输出结果。输出结果在反向传播传到量化部分时，因为量化是一个不可微分的操作，需要使用直通估计器解决。但是每一层的output受量化的weight和input的影响产生误差，这个output作为下一层的input也被量化从浮点域转换到定点域，继续影响下一层的概率分布。如此下去导致整个网络的输出和原先浮点输出相比差距极大。但是反向传播参数更新的时候，依然在浮点权重上更新，对于复杂网络优化难度极大。. Google在2018年在CVPR上提出一种新范式，训练时将input和weight的float值量化为int值，再反量化为float值，再卷积得到输出结果，这样会引入量化误差，在训练时会自动调优这个量化误差，训练效果会很好。推理时将input和weight的float值量化到int，再卷积得到结果output，注意需要对output进行移位定点运算，再与下一层进行计算。. ### 公告.

详细信息内容限制为 2000 个 token: ![博客园logo](//assets.cnblogs.com/logo.svg)
![搜索](//assets.cnblogs.com/icons/search.svg)
![搜索](//assets.cnblogs.com/icons/enter.svg)
![搜索](//assets.cnblogs.com/icons/search.svg)
![搜索](//assets.cnblogs.com/icons/search.svg)
![写随笔](//assets.cnblogs.com/icons/newpost.svg)
![我的博客](//assets.cnblogs.com/icons/myblog.svg)
![短消息](//assets.cnblogs.com/icons/message.svg)
![简洁模式](//assets.cnblogs.com/icons/lite-mode-on.svg)
![用户头像](//assets.cnblogs.com/icons/avatar-default.svg)

# [模型压缩-模型蒸馏、模型剪枝、模型量化](https://www.cnblogs.com/yqw0710/p/18348057 "发布于 2024-08-07 23:17")

一、模型蒸馏

1.1 蒸馏简介

　　知识蒸馏是指通过教师模型指导学生模型训练，通过蒸馏的方式让学生模型学习到教师模型的知识，最终使学生模型达到或媲美教师模型的准确度。

　　在模型压缩中，教师模型是一个预训练好的复杂的模型，而学生模型是一个规模较小的模型。如分类任务中，由训练好的教师模型在相同的数据下，通过将教师模型对样本的预测值作为学生模型的预测目标，指导学生模型学习，这个预测值一般指教师网络输出的类概率。教师模型的参数规模大，能够表达更好的泛化能力，学生模型的参数规模比较小，如果用通常的方法直接训练，往往达不到教师模型的泛化能力，所以通过教师模型的指导，让学生模型学习教师模型的泛化能力，以达到或媲美教师模型的精准度。

1.2 知识种类

1.3 蒸馏分类

1.4 蒸馏架构

学生网络一般是：

　　1.教师网络的简化版本，具有较少的层和每层中较少的信道。

　　2.教师网络的量化版本，其中网络结构被保留。

　　3.具有高效基本操作的小型网络。

　　4.具有优化的整体网络结构的小型网络。

　　5.与教师相同的网络。

1.5 蒸馏算法  
　　为了改进在更复杂的环境中传递知识的过程，已经出现许多不同的知识蒸馏算法。

1.6 蒸馏流程

　　1.训练教师模型。

　　2.在高温T下，教师网络产生soft lables，学生网络产生soft prediction，同时在T=1下学生网络产生hard prediction。

　　3.用步骤二生成的soft lables和soft prediction计算 distillation loss，用hard prediction和hard labley计算student loss，用这两个损失同时训练学生网络。

　　4.设置T=1，用学生网络做线上推理。

二、模型剪枝

2.1 剪枝简介

　　过参数化主要是指在训阶段，在数学上需要进行大量的微分求解，去捕捉数据中的微小的变化信息，一旦完成迭代式的训练之后，网络模型在推理的时候不需要这么多参数，而剪枝算法正是基于过参数化的理论基础提出来的。剪枝算法核心思想就是减少网络模型中的参数量和计算量，同时尽量保证模型的性能不受影响。

2.2 剪枝步骤   
　　对模型剪枝有三种常见做法：  
　　1.训练一个模型->对模型进行剪枝->对剪枝后的模型进行微调（最常见）；  
　　2.在模型训练过程中进行剪枝->对剪枝后的模型进行微调；  
　　3.进行剪枝->从头训练剪枝后的模型。  
　　剪枝可以进行细粒度剪枝、向量剪枝、核剪枝、滤波器剪枝等不同的剪枝算法。其中很重要的一点是在剪枝之后，对网络模型进行评估，看是否符合要求。剪枝之前需要确定需要剪枝的层，设定一个剪枝阈值或者比例，在具体实现上，通过修改代码，加入一个与参数矩阵尺寸一致的mask矩阵，mask矩阵中只有0和1，它的实际作用是微调网络。  
　　微调是恢复被剪枝操作影响的模型表达能力的必要步骤，结构化剪枝会对原始模型的结构进行调整，因此剪枝后的模型虽然保留了原始模型的参数，但是由于模型结构的改变，模型的表达能力会受到一定程度的影响。具体实现上，在计算的时候先乘以该mask矩阵，mask为1的值将继续训练，并通过反向传播调整梯度，而mask为0的部分因为输出始终为0，则不对后面产生影响。

2.3 剪枝分类

2.3.1 结构化剪枝与非结构化剪枝。非结构化剪枝粒度最小，结构化剪枝中的层级、通道级、滤波器级剪枝粒度依次增大。

2.3.2 静态剪枝与动态剪枝。静态剪枝在推理之前离线执行所有剪枝步骤，而动态剪枝在运行时执行。

2.3.3 硬剪枝与软剪枝。都是对filter进行剪枝，是结构化剪枝中粒度最大的剪枝。

三、模型量化

3.1 量化简介

       数字精度（如32位浮点数、16位浮点数或8位浮点数或8位整数），所能表示的范围不同。不同的数字精度会影响模型大小和推理时间，范围越大，精度越高，模型越大，推理时间越长。

　　卷积神经网络特点：参数量大，计算量大，内存占用多，精度高。

　　模型量化就是把高位宽（float32）表示的权值或者激活值用较低位宽来近似表示（int8，int4....）在数值上的体现就是将连续的值离散化。模型量化，降低精度以减少模型尺寸，会存在一定的损失，但有时候需要快速的做输出，并不一定要那么精确。模型量化特点：压缩参数，提升速度，降低内存占用，可接受的精度损失。

　　量化有两种，一种是在训练当中进行降低，一种是预训练完成后再降低。一般第二种用的多。在huggingface里面，模型量化分两种，第一种是GGML，为了在CPU上更快更好的训练模型，对CPU推出一种模型量化方式。第二种GPTQ对GPU推出一种量化方式。

3.2 量化优点

　　1.减小模型大小

　　2.降低访存，执行神经网络时大部分时间都用于访存(如取指，访存，写回的时间较长)，但在执行神经网络在gpu上的时间很少，量化能降低访存时间。

　　3.加快速度，gpu在Pascal架构下支持dp4a指令，可以一次完成4个8比特整型的乘加，Xilinx的DSP框架也会对整型运算加速。

3.3 量化分类

3.4 量化方法

　　早期量化，将input和weight进行量化，再卷积得到输出结果。输出结果在反向传播传到量化部分时，因为量化是一个不可微分的操作，需要使用直通估计器解决。但是每一层的output受量化的weight和input的影响产生误差，这个output作为下一层的input也被量化从浮点域转换到定点域，继续影响下一层的概率分布。如此下去导致整个网络的输出和原先浮点输出相比差距极大。但是反向传播参数更新的时候，依然在浮点权重上更新，对于复杂网络优化难度极大。

　　Google在2018年在CVPR上提出一种新范式，训练时将input和weight的float值量化为int值，再反量化为float值，再卷积得到输出结果，这样会引入量化误差，在训练时会自动调优这个量化误差，训练效果会很好。推理时将input和weight的float值量化到int，再卷积得到结果output，注意需要对output进行移位定点运算，再与下一层进行计算。  
　　BN折叠量化，BN层折叠到卷积层，折叠进去之后也可以量化。  
　　ReLU折叠量化，也可以折叠，直接进行量化。  
　　Add量化，与训练代码无关，所以可以直接在推理框架里量化。  
　　Concat量化等。

![](https://img2024.cnblogs.com/blog/35695/202604/35695-20260423213336272-1914399152.webp)

### 公告

![](//assets.cnblogs.com/images/ghs.png)

信息来源: \N

URL: https://developer.aliyun.com/article/1607815

信息内容: 本文系统解析深度学习模型压缩三大核心技术：剪枝、量化与知识蒸馏，详解如何实现模型缩小16倍、推理加速4倍。涵盖技术原理、工程实践与组合策略，助力AI模型

详细信息内容限制为 2000 个 token: ![](https://img.alicdn.com/imgextra/i2/O1CN01bYc1m81RrcSAyOjMu_!!6000000002165-54-tps-60-60.apng)

### 探索云世界

#### 热门

#### [云计算](https://developer.aliyun.com/ecs/)

#### [大数据](https://developer.aliyun.com/bigdata/)

#### [云原生](https://developer.aliyun.com/cloudnative/)

#### [人工智能](https://developer.aliyun.com/modelscope/)

#### [数据库](https://developer.aliyun.com/database/)

#### [开发与运维](https://developer.aliyun.com/group/othertech/)

### 活动广场

丰富的线上&线下活动，深入探索云世界

#### 任务中心

做任务，得社区积分和周边

#### 训练营

资深技术专家手把手带教

#### 直播

技术交流，直击现场

#### 乘风者计划

让创作激发创新

### 下载

海量开发者使用工具、手册，免费下载

#### 镜像站

极速、全面、稳定、安全的开源镜像

#### 技术资料

开发手册、白皮书、案例集等实战精华

热门

# 深度学习中的模型压缩技术：从理论到实践

深度学习在过去十年中取得了巨大的进步，但伴随着这些进步的是模型变得越来越庞大和复杂。这引发了对模型压缩技术的需求，旨在减小模型大小、加速推理速度并降低计算成本。本文将详细介绍几种主流的模型压缩技术，并通过实际案例来分析它们的有效性和应用场景。  
一、模型压缩的理论基础  
在深入探讨具体的模型压缩技术之前，我们需要了解其背后的基本理论。深度学习模型通常包含大量的参数，这些参数在训练过程中逐渐调整以最小化损失函数。然而，并非所有的参数都是同等重要的。一些参数可能对模型的预测能力几乎没有贡献，这就为模型压缩提供了可能性。  
二、常见的模型压缩方法

![](https://ucc.alicdn.com/avatar/7v53mftipku2u_39629444b41a4f998c9fdab985826d4c.jpg?x-oss-process=image/resize,h_150,m_lfit)

### 为什么选择阿里云

### 大模型

### 产品和定价

### 技术内容

### 权益

### 服务

### 关注阿里云

关注阿里云公众号或下载阿里云APP，关注云资讯，随时随地运维管控云服务

![阿里云APP](https://img.alicdn.com/imgextra/i4/O1CN01XLesV31fkf7pYNATb_!!6000000004045-2-tps-400-400.png)
![阿里云微信](https://img.alicdn.com/tfs/TB1AOdINW6qK1RjSZFmXXX0PFXa-258-258.jpg)

联系我们：4008013260

### 友情链接

© 2009-现在 Aliyun.com 版权所有 增值电信业务经营许可证： [浙B2-20080101](http://beian.miit.gov.cn/) 域名注册服务机构许可： [浙D3-20210002](https://domain.miit.gov.cn/域名注册服务机构/互联网域名/阿里云计算有限公司 )

[![](//gw.alicdn.com/tfs/TB1GxwdSXXXXXa.aXXXXXXXXXXX-65-70.gif)](https://zzlz.gsxt.gov.cn/businessCheck/verifKey.do?showType=p&serial=91330106673959654P-SAIC_SHOW_10000091330106673959654P1710919400712&signData=MEUCIQDEkCd8cK7%2Fyqe6BNMWvoMPtAnsgKa7FZetfPkjZMsvhAIgOX1G9YC6FKyndE7o7hL0KaBVn4f%20V%2Fiof3iAgpsV09o%3D)[![浙公网安备 33010602009975号](//img.alicdn.com/tfs/TB1..50QpXXXXX7XpXXXXXXXXXX-40-40.png)浙公网安备 33010602009975号](http://www.beian.gov.cn/portal/registerSystemInfo)[浙B2-20080101-4](https://beian.miit.gov.cn/)

![](//gw.alicdn.com/tfs/TB1GxwdSXXXXXa.aXXXXXXXXXXX-65-70.gif)
![浙公网安备 33010602009975号](//img.alicdn.com/tfs/TB1..50QpXXXXX7XpXXXXXXXXXX-40-40.png)

信息来源: 一文带你了解模型量化、剪枝和蒸馏 - 53AI-AI知识库|企业AI知识库|大模型知识库|AIHub

URL: https://www.53ai.com/news/LargeLanguageModel/2025101338470.html

信息内容: 大模型技术 多模态技术 RAG技术 知识图谱 模型微调 Skill 提示词技巧 开源大模型 智能硬件 Palantir. langchain llamaindex RAGFlow coze Dify Fastgpt Bisheng Qanything MaxKB Openclaw. AI+汽车 AI+金融 AI+工业 AI+培训 AI+SaaS AI+电商 AI+医疗. 训练后量化（Post-Training Quantization, PTQ）**. 量化感知训练（Quantization-Aware Training, QAT）**. * 优点：压缩率高（可移除 50%-90% 参数）；缺点：稀疏矩阵难以被硬件加速（普通 GPU/CPU 对非连续内存访问效率低）。. * 按 “结构单元” 移除冗余（如 CNN 的整个卷积核、通道，Transformer 的整个注意力头），保留模型的密集性。. * 优点：适配硬件加速（如 GPU 的卷积计算优化），部署友好；缺点：压缩率略低（通常移除 30%-60% 参数）。. 让学生模型的中间层特征（如 CNN 的卷积层输出、Transformer 的隐藏状态）模仿教师模型的对应层特征，保留更深层的任务相关信息。. | 量化 | 降低参数精度 | 实现简单，硬件加速友好 | 过低精度可能导致性能下降 | 剪枝 + 量化（先精简结构，再降精度） |. | 剪枝 | 移除冗余参数 / 结构 | 直接减少计算量和参数数量 | 需精细调参避免性能损失 | 蒸馏 + 剪枝（用教师指导剪枝后的学生） |. | 蒸馏 | 小模型模仿大模型 | 性能接近大模型，泛化性好 | 需要教师模型，训练流程复杂 | 量化 + 蒸馏（低精度小模型学习大模型知识） |. OpenAI GPT-5.5 即将上线 Microsoft Foundry（国际版） 2026-04-24. GPT5.5来了，最大特点解析 2026-04-24. GPT-5.5来了！我撤回了退订ChatGPT的决定 2026-04-24. GPT-5.5 发布，详细解读 2026-04-24. GPT-5.5来了！全榜第一碾压Opus 4.7，OpenAI今夜雪耻. Claude Opus 4.7刚刚曝光！Claude Code一夜重构，7x24小时替你打工. 2026-03-31 2026年 国内如何注册 Claude 账号教程. 内容创作   大模型技术   个人提效   langchain   llamaindex   多模态技术   RAG技术   智能客服   知识图谱   模型微调   RAGFlow   coze   Dify   Fastgpt   Bisheng   Qanything   AI+汽车   AI+金融   AI+工业   AI+培训   AI+SaaS   Skill   提示词技巧   AI+电商   AI面试   数字员工   ChatBI   AI知识库   开源大模型   智能营销   智能硬件   智能化改造   AI+医疗   MaxKB   Palantir   Glean   Openclaw.

详细信息内容限制为 2000 个 token: 2026年4月29日 周三晚上19:30，来了解“企业AI训练师：从个人提效到构建企业AI生产力”（限30人）

[![](https://static.53ai.com/assets/static/images/logo.png)](/) 

![](https://static.53ai.com/uploads/20240311/eb21a9e0409ab946254c64427055a5fa.png)   免费POC， 零成本试错

* ![]()

  工作+AI   
   大模型提升全员工作效率

  ![]()

  业务+AI   
   大模型掌握企业知识与流程

  ![]()

  AIx业务   
   大模型驱动产品智能化改造

* [![]()

  了解更多 >](/consulting.html)
* [![]()

  Co-creation

  AI场景共创

  了解更多 >](/fine-tuning.html)

热门产品

[![53AI Brain](https://static.53ai.com/uploads/20250429/f9ed1b6c2d16a688c5791a0057c2217d.png)

53AI Brain

让知识在人与AI之间高效流动](/products/53AIBrain) [![53AI Studio](https://static.53ai.com/uploads/20250429/b2cd4330d03bce8eb0eb0332eb2954cd.png)

53AI Studio

高准确率的企业级智能体开发平台](/products/53AIStudio) [![53AI Hub](https://static.53ai.com/uploads/20250429/94e6d990ee63512db567bc53c113e8bd.png)

53AI Hub开源

三分钟搭建出独立的企业AI门户](/products/53AIHub) [![53AI Browser](https://static.53ai.com/uploads/20250429/05d03bb026421069826c64e2407ad7ee.png)

53AI Browser

“AI专家”效率倍增的秘密武器

敬请期待...](javascript:;)

[客户案例](/kehuanli.html)

* 行业案例

  [![政府央国企](https://static.53ai.com/uploads/20250718/14a52c018cf42a962cd782bc00deecaf.png)

  政府央国企   
   政府央国企大模型落地应用案例](/kehuanli/hangyeanli#solution-230) [![能源矿业](https://static.53ai.com/uploads/20250718/febd8ab7274bef72d77798177fa4b55a.png)

  能源矿业   
   新能源与矿业大模型落地应用案例](/kehuanli/hangyeanli#solution-231) [![电子科技](https://static.53ai.com/uploads/20250718/94f808db1b269c89e3da457d49752174.png)

  电子科技   
   电子科技行业大模型落地应用案例](/kehuanli/hangyeanli#solution-232) [![贸易流通](https://static.53ai.com/uploads/20250718/0816ed40b43b19540c8149301535a27e.png)

  贸易流通   
   贸易流通大模型落地应用案例](/kehuanli/hangyeanli#solution-235) [![制造行业](https://static.53ai.com/uploads/20250718/4c7688c517858072bee15b3eae9a8eb1.png)

  制造行业   
   高端制造行业大模型落地应用案例](/kehuanli/hangyeanli#solution-236) [![企科数服](https://static.53ai.com/uploads/20250718/20339b3fdb032d7ee0d009b2a3595191.png)

  企科数服   
   企科数服行业大模型落地应用案例](/kehuanli/hangyeanli#solution-237) [![生物医药](https://static.53ai.com/uploads/20250718/2640c0c52d0e598bf673372b5d771d4a.png)

  生物医药   
   生物医药行业大模型落地应用案例](/kehuanli/hangyeanli#solution-234) [![地产与消费品](https://static.53ai.com/uploads/20250718/f3e87229b4498063145a2c622d2072cc.png)

  地产与消费品   
   地产与消费品行业大模型落地应用案例](/kehuanli/hangyeanli#solution-233)

* 场景案例

  [![【智能问答】场景案例](https://static.53ai.com/assets/static/images/solution_icon_1.png)

  【智能问答】场景案例   
   让大模型掌握企业的知识和流程](/kehuanli/solution#solution-148) [![【应用智改】场景案例](https://static.53ai.com/assets/static/images/solution_icon_2.png)

  【应用智改】场景案例   
   让大模型融入企业的产品和业务](/kehuanli/solution#solution-149) [![【智能工单】场景案例](https://static.53ai.com/assets/static/images/solution_icon_3.png)

  【智能工单】场景案例   
   让大模型创建和受理业务工单](/kehuanli/solution#solution-150) [![【智能问数】场景案例](https://static.53ai.com/uploads/20250811/9be0a71f6e9b51b9cd9ca9b7dfeddd95.png)

  【智能问数】场景案例   
   与业务系统数据对话式互动](/kehuanli/solution#solution-151)

[AI知识库](/news.html)

企业AI落地知识库

[前沿技术](/news/qianyanjishu)

[大模型技术](/news/LargeLanguageModel) [多模态技术](/news/MultimodalLargeModel) [RAG技术](/news/RAG) [知识图谱](/news/knowledgegraph) [模型微调](/news/finetuning) [Skill](/news/tishicikuangjia) [提示词技巧](/news/tishicijiqiao) [开源大模型](/news/OpenSourceLLM) [智能硬件](/news/zhinengyingjian) [Palantir](/news/Palantir)

[Agent框架](/news/agentplatform)

[langchain](/news/langchain) [llamaindex](/news/llamaindex) [RAGFlow](/news/RAGFlow) [coze](/news/coze) [Dify](/news/dify) [Fastgpt](/news/fastgpt) [Bisheng](/news/Bisheng) [Qanything](/news/Qanything) [MaxKB](/news/MaxKB) [Openclaw](/news/Openclaw)

[行业应用](/news/hangyeyingyong)

[AI+汽车](/news/AIqiche) [AI+金融](/news/AIjinrong) [AI+工业](/news/AIgongye) [AI+培训](/news/AIpeixun) [AI+SaaS](/news/AISaaS) [AI+电商](/news/AIdianshang) [AI+医疗](/news/AIyiliao)

[企业落地](/news/qiyejingying)

[内容创作](/news/neirongchuangzuo) [个人提效](/news/gerentixiao) [智能客服](/news/zhinengkefu) [AI面试](/news/AImianshi) [数字员工](/news/shuziyuangong) [ChatBI](/news/zhinengbaobiao) [AI知识库](/news/zhishiguanli) [智能营销](/news/zhinengyingxiao) [智能化改造](/news/zhinenghuagaizao) [Glean](/news/Glean)

[行业报告](/hangyebaogao.html)

[研究报告](/hangyebaogao.html?report_type=研究报告) [行业报告](/hangyebaogao.html?report_type=行业报告) [技术分享](/hangyebaogao.html?report_type=技术分享) [专题报告](/hangyebaogao.html?report_type=专题报告) [课件讲义](/hangyebaogao.html?report_type=课件讲义)

[关于我们](/about.html)

[公司介绍](/about/introduction)  [渠道合作](/about/cooperation)

[GitHub Star 5.7K+](https://github.com/53ai/53aihub)  [预约演示](/trial.html)

![]()

![]()

* [首页](/)
* [产品服务](javascript:)
* [客户案例](javascript:)
* [AI知识库](javascript:)
* [关于我们](javascript:)

热门场景

![]()

工作+AI

[![]()

工作对话](/product/gongzuoduihua)  [![]()

内容创作](/product/neirongchuangzuo)  [![]()

方案撰写](/product/zhinengwendang)  [![]()

魔法菜单](/product/mofacaidan)

![]()

业务+AI

[![]()

微信分身](/product/weixinfenshen)  [![]()

海外客服](/product/haiwaikefu)  [![]()

官网客服](/product/guanwangkefu)  [![]()

抖音客服](/product/douyinkefu)  [![]()

数字老师](/product/shuzilaoshi)  [![]()

数字督导](/product/shuzidudao)  [![]()

智能服务台](/product/zhinengfuwutai)

![]()

AIx业务

[![]()

智能问数](/product/zhinengwenshu)  [![]()

智能审核](/product/zhinengshenhe)  [![]()

智能工单](/product/zhinenggongdan)  [![]()

企微跟进助手](/product/qiweigenjinzhushou)  [![]()

智能报价](/product/zhinengbaojia)  [![]()

企微销售助手](/product/qiweixiaoshouzhushou)  [![]()

应用智改](/product/zijianyingyong)  [![]()

企微客服助手](/product/qiweikefuzhushou)

[落地咨询](/consulting.html)

[场景共创](/fine-tuning.html)

热门产品

[![53AI Brain](https://static.53ai.com/uploads/20250429/f9ed1b6c2d16a688c5791a0057c2217d.png)

53AI Brain

让知识在人与AI之间高效流动](/products/53AIBrain) [![53AI Studio](https://static.53ai.com/uploads/20250429/b2cd4330d03bce8eb0eb0332eb2954cd.png)

53AI Studio

高准确率的企业级智能体开发平台](/products/53AIStudio) [![53AI Hub](https://static.53ai.com/uploads/20250429/94e6d990ee63512db567bc53c113e8bd.png)

53AI Hub开源

三分钟搭建出独立的企业AI门户](/products/53AIHub) [![53AI Browser](https://static.53ai.com/uploads/20250429/05d03bb026421069826c64e2407ad7ee.png)

53AI Browser

“AI专家”效率倍增的秘密武器

敬请期待...](javascript:;)

[行业案例](/kehuanli/hangyeanli) [场景案例](/kehuanli/solution)

[前沿技术](/news/qianyanjishu) [Agent框架](/news/agentplatform) [行业应用](/news/hangyeyingyong) [企业落地](/news/qiyejingying)

[公司介绍](/about/introduction) [渠道合作](/about/cooperation)

![AI知识库](https://static.53ai.com/uploads/20250210/aec076a60258b0cc07078c8ea7dff92c.webp)

53AI知识库

学习大模型的前沿技术与行业应用场景

[立即咨询](javascript:;) [预约演示](javascript:;)

[![](https://static.53ai.com/assets/static/images/tab1.png) 首页](/)   [AI知识库](/news.html)   [前沿技术](/news/qianyanjishu)   [大模型技术](/news/LargeLanguageModel)

![](https://static.53ai.com/assets/static/images/edit-icon.png)

我要投稿

# 一文带你了解模型量化、剪枝和蒸馏

发布日期：2025-10-13 14:16:17 浏览次数： 2489

作者：华为云开发者联盟

![](https://static.53ai.com/assets/static/images/detail-icon.png)

微信搜一搜，关注“华为云开发者联盟”

推荐语

模型压缩技术揭秘：量化、剪枝、蒸馏三大法宝，让AI模型在资源受限设备上高效运行！ 核心内容： 1. 模型量化的原理与方法：降低参数精度，减少存储与计算成本 2. 模型剪枝的技术分类：结构化与非结构化剪枝的优缺点对比 3. 模型蒸馏的核心思想：用大模型指导小模型训练，实现知识迁移

![](https://static.53ai.com/assets/static/images/avatar.jpg)

杨芳贤

53AI创始人/腾讯云(TVP)最具价值专家

![图片](https://api.ibos.cn/v4/weapparticle/accesswximg?aid=125625&url=aHR0cHM6Ly9tbWJpei5xcGljLmNuL3N6X21tYml6X3BuZy8xYWR5dnoxUmJRODFpYjZtWU9meUo0Nmp1UHpwSFo0R3Z0WWtZS0NCeHlEYXVTMG94cldabkFpYlpGOXRIVXZNVFJwUlFjdW1hQ2tuTnNOWHRFRldTcmp3LzY0MD93eGZyb209NSZhbXA=;wx_lazy=1&tp=webp#imgIndex=0)

模型量化、剪枝和蒸馏是三种主流的模型压缩与优化技术，核心目标是在保证模型性能（精度、准确率）的前提下，减小模型体积、降低计算复杂度，使其能在资源受限的设备（如手机、嵌入式设备、边缘终端）上高效部署。

***一*****模型量化（Model Quantization）**

**降低参数精度，减少存储与计算成本**

**核心的原理**

将模型中高精度的参数（如 32 位浮点数，FP32）转换为低精度格式（如 16 位浮点数 FP16、8 位整数 INT8，甚至 4 位、2 位、1 位），利用神经网络对 “噪声” 的容忍性，在精度损失可控的前提下，减少参数存储量和计算量。

**关键方法**

**1. 训练后量化（Post-Training Quantization, PTQ）**

* 直接对训练好的模型参数进行量化，无需重新训练，操作简单（如 TensorFlow Lite 的量化工具）。
* 缺点：精度损失可能较大（尤其低至 INT8 以下时），适合对精度要求不高的场景（如简单图像分类）。

**2. 量化感知训练（Quantization-Aware Training, QAT）**

* 在训练过程中模拟低精度量化的误差（如数值截断、舍入），让模型 “适应” 量化带来的噪声，最终输出量化模型。
* 优点：精度损失小（INT8 量化可保留原模型 95% 以上性能），适合高精度需求场景（如目标检测、医学影像）。

**效果与适用场景**

* **压缩效果：FP32→INT8 可减少 75% 存储量（32 位→8 位），计算速... [truncated]

信息来源: 大模型量化、蒸馏、剪枝：2026年模型压缩技术完全指南 - 稀土掘金

URL: https://juejin.cn/post/7629971984269705242

信息内容: `FP32 (4 bytes) → FP16 (2 bytes) → BF16 (2 bytes). → INT8 (1 byte) → INT4 (0.5 byte) → INT2 (0.25 byte)`. | FP16 | 极小 | 1.5-2x | 50% | 显存够但想省钱 |. | BF16 | 极小 | 1.5-2x | 50% | A100/H100 最优选 |. | INT8 (LLM.int8) | 小 | 2-3x | 75% | 均衡选择 |. | GPTQ-INT4 | 中 | 3-4x | 87.5% | 消费级 GPU 首选 |. | AWQ-INT4 | 小（优于GPTQ） | 3-4x | 87.5% | 2026年推荐方案 |. ### 推荐实践：AWQ 量化 Qwen3-7B. quant_path = "Qwen3-7B-AWQ-INT4". 量化后：Qwen3-7B FP16 (14GB) → AWQ-INT4 (4.2GB)，速度提升约 3.5x。. **坑1**：INT4 对注意力层量化更敏感，建议对 lm\_head 和 embed\_tokens 保持 FP16。. `传统训练：Student 学 {输入→正确标签}. 知识蒸馏：Student 学 {输入→Teacher的输出概率分布}`. 典型案例：用 GPT-5 生成 100 万条高质量对话，训练 7B Student 模型。. "output": response.choices[0].message.content. 访问 Teacher 的中间层激活值，让 Student 的中间层也对齐。. 精度损失最小，但需要 Teacher 开源。适合 GLM-5.1 → GLM-3.5 这类同系列蒸馏。. outputs = model(**batch, output_attentions=True). layer.attention.self.query.weight.grad. * layer.attention.self.query.weight. **方案A（最大压缩）**：剪枝30% → 蒸馏恢复精度 → AWQ-INT4 量化. **方案B（均衡）**：AWQ-INT4 量化 + CoT 蒸馏. **方案C（轻度压缩）**：AWQ-INT8 量化. ## 五、2026 年 OCR 大模型的量化实践案例. 某 OCR 大模型团队在 2026 奇点大会公布了以下数据：.

详细信息内容限制为 2000 个 token: ![稀土掘金](//lf-web-assets.juejin.cn/obj/juejin-web/xitu_juejin_web/e08da34488b114bd4c665ba2fa520a31.svg)
![稀土掘金](//lf-web-assets.juejin.cn/obj/juejin-web/xitu_juejin_web/6c61ae65d1c41ae8221a670fa32d05aa.svg)

# 大模型量化、蒸馏、剪枝：2026年模型压缩技术完全指南

## 为什么模型压缩在 2026 年比以往更重要？

GPT-5、Claude Opus 4、GLM-5.1 这些顶级模型能力越来越强，但参数量也越来越大。在以下场景中，"把大模型搬到生产环境"成了真实挑战：

本文系统梳理量化（Quantization）、知识蒸馏（Distillation）、剪枝（Pruning）三大压缩技术，以及 2026 年的最新实践。

## 一、量化（Quantization）：用更少的比特表示权重

### 核心原理

模型权重默认用 float32（32位）或 float16（16位）存储。量化就是把这些精度降低：

`FP32 (4 bytes) → FP16 (2 bytes) → BF16 (2 bytes)
→ INT8 (1 byte) → INT4 (0.5 byte) → INT2 (0.25 byte)`

模型大小线性降低，推理速度大幅提升，精度有损但可控。

### 2026 年主流量化方案对比

| 方案 | 精度损失 | 速度提升 | 内存节省 | 适用场景 |
| --- | --- | --- | --- | --- |
| FP16 | 极小 | 1.5-2x | 50% | 显存够但想省钱 |
| BF16 | 极小 | 1.5-2x | 50% | A100/H100 最优选 |
| INT8 (LLM.int8) | 小 | 2-3x | 75% | 均衡选择 |
| GPTQ-INT4 | 中 | 3-4x | 87.5% | 消费级 GPU 首选 |
| AWQ-INT4 | 小（优于GPTQ） | 3-4x | 87.5% | 2026年推荐方案 |
| GGUF-Q4\_K\_M | 中 | 3-4x | 约80% | CPU 推理/本地部署 |

### 推荐实践：AWQ 量化 Qwen3-7B

`from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model_path = "Qwen/Qwen3-7B"
quant_path = "Qwen3-7B-AWQ-INT4"
# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 量化配置
quant_config = {
"zero_point": True,
"q_group_size": 128,
"w_bit": 4,
"version": "GEMM"
}
# 执行量化（需要校准数据集）
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)`

量化后：Qwen3-7B FP16 (14GB) → AWQ-INT4 (4.2GB)，速度提升约 3.5x。

### 量化踩坑记录

**坑1**：INT4 对注意力层量化更敏感，建议对 lm\_head 和 embed\_tokens 保持 FP16。

**坑2**：GPTQ 量化需要 GPU，AWQ 在量化质量上普遍优于 GPTQ，推荐优先用 AWQ。

**坑3**：量化后要用相同的评测基准测一遍，精度损失 > 5% 时需要降低压缩比。

## 二、知识蒸馏（Knowledge Distillation）：让小模型学大模型的"思维方式"

### 核心原理

蒸馏的目标：让小模型（Student）模仿大模型（Teacher）的行为，而不只是模仿训练数据的标签。

`传统训练：Student 学 {输入→正确标签}
知识蒸馏：Student 学 {输入→Teacher的输出概率分布}`

为什么概率分布比标签更有价值？

因为 Teacher 的 softmax 输出包含了"这个词和那个词有多相似"的信息。比如"猫"这个词，Teacher 可能输出 {猫:0.8, 狗:0.1, 宠物:0.06, ...}，这比简单的 one-hot 标签包含更多信息。

### 2026 年蒸馏的三种范式

#### 范式一：黑盒蒸馏（数据生成式）

无需访问 Teacher 内部结构，只需用 Teacher 生成大量高质量数据，然后训练 Student。

适用场景：Teacher 是闭源 API（如 GPT-5），你只有输出权限。

典型案例：用 GPT-5 生成 100 万条高质量对话，训练 7B Student 模型。

`# 用 Teacher API 生成训练数据
from openai import OpenAI
client = OpenAI()
training_data = []
for prompt in prompts:
response = client.chat.completions.create(
model="gpt-5",
messages=[{"role": "user", "content": prompt}],
temperature=0.7
)
training_data.append({
"input": prompt,
"output": response.choices[0].message.content
})`

#### 范式二：白盒蒸馏（中间层对齐）

访问 Teacher 的中间层激活值，让 Student 的中间层也对齐。

精度损失最小，但需要 Teacher 开源。适合 GLM-5.1 → GLM-3.5 这类同系列蒸馏。

#### 范式三：推理链蒸馏（Chain-of-Thought Distillation）

让 Teacher 生成详细的思维链（CoT），Student 不只学答案，还学推理过程。

2025-2026 年最流行的蒸馏方式，显著提升 Student 在复杂推理任务上的能力。

`Teacher 输出：
"首先分析题目条件...然后列方程...解方程得x=5...因此答案是5"
Student 学习内容：完整推理链 + 最终答案`

**效果数据**：DeepSeek-R1 就是通过 CoT 蒸馏，用 7B 模型复现了 671B 模型约 85% 的数学推理能力。

## 三、剪枝（Pruning）：删掉不重要的神经元

### 核心思路

神经网络中，并非所有参数都同等重要。剪枝通过识别并移除"不重要"的权重来缩小模型。

### 结构化 vs 非结构化

| 类型 | 方法 | 速度提升 | 实现难度 |
| --- | --- | --- | --- |
| 非结构化剪枝 | 置零单个权重 | 低（需稀疏计算加速硬件） | 低 |
| 结构化剪枝 | 移除整个注意力头或FFN神经元 | 高（标准硬件即可加速） | 中 |

2026 年推荐优先使用**结构化剪枝**，因为它在标准 GPU 上就能实现真正的推理加速。

### 注意力头剪枝实践

研究发现，大模型中约 30-40% 的注意力头是"冗余的"（对最终输出影响极小）。

`# 识别重要性低的注意力头（基于梯度信息）
import torch
def compute_head_importance(model, dataloader):
head_importance = torch.zeros(
model.config.num_hidden_layers,
model.config.num_attention_heads
)
for batch in dataloader:
outputs = model(**batch, output_attentions=True)
loss = outputs.loss
loss.backward()
for layer_idx, layer in enumerate(model.encoder.layer):
# 使用梯度×权重作为重要性估计
head_importance[layer_idx] += (
layer.attention.self.query.weight.grad
* layer.attention.self.query.weight
).abs().sum(dim=0)
return head_importance`

实测：对 7B 模型剪掉 30% 的注意力头后，推理速度提升 25%，MMLU 精度下降 < 2%。

## 四、组合策略：量化 + 蒸馏 + 剪枝的最优配比

在资源有限的情况下，如何组合使用这三种技术？

### 推荐组合方案

**方案A（最大压缩）**：剪枝30% → 蒸馏恢复精度 → AWQ-INT4 量化

**方案B（均衡）**：AWQ-INT4 量化 + CoT 蒸馏

**方案C（轻度压缩）**：AWQ-INT8 量化

## 五、2026 年 OCR 大模型的量化实践案例

某 OCR 大模型团队在 2026 奇点大会公布了以下数据：

采用 **8层量化蒸馏架构**（量化 + 层级蒸馏）后：

这个案例说明：对于精度要求在 97-99% 的应用场景，激进的量化压缩完全可以在生产环境使用。

## 总结

| 技术 | 核心价值 | 推荐场景 |
| --- | --- | --- |
| 量化 | 最易上手，效果立竿见影 | 所有需要降成本/提速的场景 |
| 蒸馏 | 精度损失最小，效果持久 | 有条件微调的团队 |
| 剪枝 | 真正减少计算量 | 边缘推理、极致压缩需求 |

2026年，模型压缩已经是 MLOps 工程师的必备技能。不会压缩模型，就像厨师不会控火——能做菜，但做不好。

![avatar](https://p26-passport.byteacctimg.com/img/user-avatar/6de388438593cd4d3c4b395aec0d8e41~200x200.image)
![创作等级LV.3](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA8AQMAAAAAMksxAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAA5JREFUKM9jGAWjAAcAAAIcAAE27nY6AAAAAElFTkSuQmCC "创作等级LV.3")

信息来源: 知识蒸馏、轻量化模型架构、剪枝…几种深度学习模型压缩方法- 知乎

URL: https://zhuanlan.zhihu.com/p/613598877

信息内容: 本文介绍了卷积神经网络常见的几种压缩方法。 按照压缩过程对网络结构的 ... 向量剪枝(vector-level)：它相对于细粒度剪枝粒度更大，属于对卷积核内部(intra-

详细信息内容限制为 2000 个 token: