# Deep Learning｜深度学习交互课程

本仓库以 Goodfellow、Bengio 与 Courville 的《Deep Learning》为主线，整理数学基础、深度网络实践与研究主题。首批网页聚焦第 6 章“深度前馈网络”，把概念解释、MathML 公式、独立 SVG、交互实验和代码示例组织成逐篇阅读的课程。

**课程网站：** https://1998x-stack.github.io/Deep-Learning/  
**第一篇：** [从 XOR 看懂前馈神经网络](https://1998x-stack.github.io/Deep-Learning/articles/xor/)

## 第一批 8 篇

| 顺序 | 模块 | 核心问题 |
|---:|---|---|
| 1 | [XOR 与前馈网络](site/articles/xor/index.html) | 为什么单条直线不够？ |
| 2 | [梯度下降](site/articles/gradient/index.html) | 参数如何更新？ |
| 3 | [损失函数](site/articles/cost-functions/index.html) | 模型究竟优化什么？ |
| 4 | [输出单元](site/articles/output-units/index.html) | 如何把结果变成任务答案？ |
| 5 | [隐藏单元](site/articles/hidden-units/index.html) | 网络如何学习特征？ |
| 6 | [ReLU](site/articles/relu/index.html) | 分段线性函数为何有效？ |
| 7 | [Sigmoid 与 Tanh](site/articles/sigmoid-tanh/index.html) | 饱和如何影响梯度？ |
| 8 | [其他隐藏单元](site/articles/other-activations/index.html) | Softplus、SELU、Swish、GELU、Mish、Maxout 如何取舍？ |

每篇至少 8 个独立 SVG、8 组 MathML 公式、一个交互实验、可运行代码与 evidence.json 证据账本。

## 仓库结构

- 1_应用数学与机器学习基础：线性代数、概率、数值计算与机器学习基础。
- 2_深度网络：现代实践：深度前馈网络、优化、正则化、卷积与序列建模。
- 3_深度学习研究：表示学习、概率模型、近似推断与生成模型。
- site：GitHub Pages 静态站点、文章、共享 CSS/JS、SVG 与证据数据。
- scripts：站点静态验证和浏览器 QA。
- tests：首批 Python 示例和站点回归测试。
- docs：站点架构、内容规范与后续路线。

## 本地预览

~~~bash
python -m http.server 8000 --directory site
# http://127.0.0.1:8000
~~~

## 验证

~~~bash
python scripts/validate_site.py
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/browser_qa.py
~~~

## GitHub Pages

.github/workflows/pages.yml 在 main 更新后执行静态验证、单元测试和 Chromium QA，再发布 site/。首次发布需要在仓库 Settings → Pages → Build and deployment → Source 选择 GitHub Actions；工作流文件本身不能替代该仓库设置。

## 内容原则

1. 先核对教材或原始论文，再写解释和画图。
2. 每张图回答一个具体问题，SVG 可独立打开并带 title/desc。
3. 数学表达使用原生 MathML，不依赖 CDN。
4. 区分教材结论、论文结果、数学推导和教学示例。
5. 每篇维护 evidence.json，重要事实要有来源和定位。
6. 借鉴渐进式叙事与图解组织，不复制 Olah/Colah 的原文、图示或品牌。

详见 AGENTS.md 与 docs/site-architecture.md。
