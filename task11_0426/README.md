# 【**任务11 -循环神经网络基础 】时长：2天**



## 1. RNN的结构。

循环神经网络的提出背景、优缺点。着重学习RNN的反向传播、RNN出现的问题（梯度问题、长期依赖问题）、BPTT算法。

提出背景：对于序列类型的数据，比如翻译问题，时间序列预测问题，由于前后样本之间有顺序之间的先后关系，而传统的神经网络无法体现这一关系，因此出现了RNN(Recurrent Neural Network)。RNN会记住之前的输入值，因此对于RNN而言，即时后面有相同的输入，输出值也会不同，这是与传统的NN不同的地方

优点：循环神经网络RNN可以对**时间序列**上有变化的情况进行处理，**神经元的输出可以在下一个时间段直接作用到自身**

缺点：RNN存在无法解决**长时依赖**（LSTM），无法并行计算（attention）

**BPTT(back-propagation through time)（RNN的反向传播）**

推导

## 2. 双向RNN

Bidirectional RNN(双向RNN)假设当前t的输出不仅仅和之前的序列有关，并且 还与之后的序列有关，例如：预测一个语句中缺失的词语那么需要根据上下文进 行预测；Bidirectional RNN是一个相对简单的RNNs，由两个RNNs上下叠加在 一起组成。输出由这两个RNNs的隐藏层的状态决定。

## 3. LSTM、GRU的结构、提出背景、优缺点。

long short term memory，即我们所称呼的LSTM，是为了解决长期以来问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。在标准RNN中，这个重复的结构模块只有一个非常简单的结构，例如一个tanh层。



LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。



不必担心这里的细节。我们会一步一步地剖析 LSTM 解析图。现在，我们先来熟悉一下图中使用的各种元素的图标。



在上面的图例中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表 pointwise 的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。

LSTM核心思想
LSTM的关键在于细胞的状态整个(绿色的图表示的是一个cell)，和穿过细胞的那条水平线。

细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。



若只有上面的那条水平线是没办法实现添加或者删除信息的。而是通过一种叫做 门（gates） 的结构来实现的。

门 可以实现选择性地让信息通过，主要是通过一个 sigmoid 的神经层 和一个逐点相乘的操作来实现的。



sigmoid 层输出（是一个向量）的每个元素都是一个在 0 和 1 之间的实数，表示让对应信息通过的权重（或者占比）。比如， 0 表示“不让任何信息通过”， 1 表示“让所有信息通过”。

LSTM通过三个这样的本结构来实现信息的保护和控制。这三个门分别输入门、遗忘门和输出门。

LSTM变体
原文这部分介绍了 LSTM 的几个变种，还有这些变形的作用。在这里我就不再写了。有兴趣的可以直接阅读原文。

下面主要讲一下其中比较著名的变种 GRU（Gated Recurrent Unit ），这是由 Cho, et al. (2014) 提出。在 GRU 中，如下图所示，只有两个门：重置门（reset gate）和更新门（update gate）。同时在这个结构中，把细胞状态和隐藏状态进行了合并。最后模型比标准的 LSTM 结构要简单，而且这个结构后来也非常流行。



其中，
 表示重置门
	
 表示更新门。重置门决定是否将之前的状态忘记。(作用相当于合并了 LSTM 中的遗忘门和传入门）当
 趋于0的时候，前一个时刻的状态信息
	
 会被忘掉，隐藏状态	
 会被重置为当前输入的信息。更新门决定是否要将隐藏状态更新为新的状态
	
 （作用相当于 LSTM 中的输出门） 。

和 LSTM 比较一下：

GRU 少一个门，同时少了细胞状态
在 LSTM 中，通过遗忘门和传入门控制信息的保留和传入；GRU 则通过重置门来控制是否要保留原来隐藏状态的信息，但是不再限制当前信息的传入。
在 LSTM 中，虽然得到了新的细胞状态 Ct，但是还不能直接输出，而是需要经过一个过滤的处理:同样，在 GRU 中, 虽然我们也得到了新的隐藏状态，但是还不能直接输出，而是通过更新门来控制最后的输出

手动推导复习GRU、LSTM

4、针对梯度消失（LSTM等其他门控RNN）、梯度爆炸（梯度截断）的解决方案。

保持cell中的状态信息，传递到下一次的运算当中

\5. Text-RNN的原理。

\6. 利用Text-RNN模型来进行文本分类。





参考资料 

1. 一份详细的LSTM和GRU图解：[一份详细的LSTM和GRU图解 -ATYUN](https://www.atyun.com/30234.html)

1. Tensorflow实战(1): 实现深层循环神经网络：[Tensorflow实战(1): 实现深层循环神经网络 - 知乎](https://zhuanlan.zhihu.com/p/37070414)

1. lstm：[从LSTM到Seq2Seq-大数据算法](https://x-algo.cn/index.php/2017/01/13/1609/)

1. RCNN kreas：[GitHub - airalcorn2/Recurrent-Convolutional-Neural...](https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier)

1. RCNN tf：[GitHub - zhangfazhan/TextRCNN: TextRCNN 文本分类](https://github.com/zhangfazhan/TextRCNN)

1. RCNN tf (推荐)：[GitHub - roomylee/rcnn-text-classification: Tensor...](https://github.com/roomylee/rcnn-text-classification)