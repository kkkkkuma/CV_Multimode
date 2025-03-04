# CV_Multimode
personal learning process
▎阶段一：核心基础强化（Week 1-2）
目标：夯实Python编程、算法与数学基础，掌握CV核心模型与PyTorch实战

1. Python进阶与算法（Days 1-7）
上午：Python高级特性

重点掌握：装饰器（@classmethod/@staticmethod）、生成器/yield、多线程（ThreadPoolExecutor）、上下文管理器（with）

实战：用生成器实现大数据批处理，用装饰器实现函数耗时统计

下午：LeetCode高频题型

每日3题：优先攻克二叉树（路径总和、最近公共祖先）、动态规划（背包问题、最长子序列）、双指针（滑动窗口）

重点刷题资源：LeetCode热题HOT 100前50题（侧重数组/字符串/链表）

晚上：数学基础回顾

线性代数：重点理解矩阵分解（SVD）、特征值与PCA降维的关系

概率统计：贝叶斯定理、常见分布（高斯/泊松）、最大似然估计推导

2. 计算机视觉基础（Days 8-14）
上午：CV核心模型精读

精读论文：ResNet、Transformer、Mask R-CNN

关键掌握：残差连接设计思想、Self-Attention计算流程、RoI Align与RoI Pooling区别

下午：PyTorch实战

手写经典模型：用PyTorch实现ResNet-18（含自定义Dataset/Dataloader）

关键技巧：模型参数冻结（requires_grad=False）、梯度裁剪（torch.nn.utils.clip_grad_norm_）

晚上：OpenCV实战

实现图像增强：混合使用仿射变换（cv2.warpAffine）、直方图均衡化（cv2.equalizeHist）

项目实战：用SIFT特征匹配实现图像拼接（参考OpenCV官方教程）

▎阶段二：多模态与项目实战（Week 3）
目标：掌握多模态核心技术，完成1-2个高质量项目

1. 多模态核心模型（Days 15-18）
上午：论文精读与代码复现

精读论文：CLIP、BLIP

关键掌握：对比学习损失（InfoNCE Loss）、图文特征对齐方法

实战：使用HuggingFace Transformers库调用CLIP模型实现零样本分类

下午：多模态融合技术

学习跨模态注意力机制（如ViLBERT中的Co-TRM层）

实战：用PyTorch实现简单的图文注意力融合（参考代码示例）

2. 项目实战（Days 19-21）
Kaggle项目：选择图像分类比赛或自建多模态项目

关键步骤：数据增强（Albumentations库）、模型集成（EMA权重平均）、结果可视化（TensorBoard）

进阶技巧：使用wandb记录超参数与实验过程

论文复现：选择一篇近期顶会多模态论文（如FLAVA）复现核心模块

重点：理解消融实验设计，记录复现过程中的性能差异

▎阶段三：面试冲刺（Week 4）
目标：模拟面试与知识体系查漏补缺

1. 高频考点梳理（Days 22-25）
计算机视觉

理论问题：解释BatchNorm在训练/推理时的差异，为什么Transformer在CV中逐渐替代CNN？

代码手撕：实现非极大值抑制（NMS）、IoU计算

多模态

理论问题：对比CLIP与传统视觉模型的优势，如何处理图文噪声对齐？

开放问题：设计一个视频问答系统（需说明特征提取与融合方案）

2. 模拟面试（Days 26-28）
行为面试：准备3个核心故事（如项目攻坚、团队协作），使用STAR法则描述

技术模拟：使用AI模拟面试工具或找同学进行Mock Interview

真题演练：参考上海AI Lab过往面试题（如牛客网面经中搜索“AI Lab实习”）

▎关键资源推荐
代码实战

CV：MMDetection（目标检测框架）

多模态：OpenCLIP（CLIP开源实现）

面试题库

机器学习：百面机器学习第5、11章

代码题：剑指Offer专项

前沿跟踪

关注上海AI Lab官网最新论文（如CVPR/ICCV收录工作）

订阅Arxiv每日推送（使用arxiv-sanity筛选cv/multimodal标签）

▎注意事项
代码规范：面试中会关注代码可读性，建议使用flake8进行格式检查

模型部署：了解ONNX/TensorRT基础概念（如模型量化），可加分

研究方向：提前阅读面试团队近期论文，面试时结合其工作提问（如“您在ECCV 2023的多模态表征工作中提到...”）
