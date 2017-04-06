---
title: Where to put the Image in an Image Caption Generator
layout: page
date: 2017-04-06
categories: 
- paper_reading
tag: paper
blog: true
start: false
author: karl
description: image_caption
--- 

在image caption这个任务中，需要输入两种特征：image, word_vector,本文就两种特征融合的位置作讨论，同时也是阅读文章[^1]之后记录一下．  

# where to put image  
关于image的位置，即与word融合的位置，在文章中[^1]中做出了详细分类,从大方向上来看, 分为两类:  
* inject  
* merging  

而如果再细分，image与word混合输入RNN的情况又可以分为四种情况：　　
* Init-inject  
* Pre-inject  
* Par-inject  
* Post-inject
接下来我们就以上集中分类进行详细介绍．　　

## injecting VS. mergeing  
对于injecting和merging的理解，可以如下图所示：　　

![1](../downloads/whereimg/1.png)  

这里需要说明，word为word_embedding vector,image为从pre-train模型最后一层fc层提取的特征，由上图可以看到，injecting更专注于word
与image的混合encode,而merging更倾向于单独对word编码，然后利用word高层表示与image进行＂融合＂．简而言之，如果image对于RNN encode
过程有作用，那么可以将其与word一起encode,反之，进行merging.　　

## Init，Pre，Par，Post  
对于inject,主要在于word与image的组织形式，而这其中基本就是近几年image caption中论文的各种创新点．主要组织形式如下图所示：　　

![2](../downloads/whereimg/2.png)  

接下来就结合各个论文做简明分析．　　

### Init-inject  
init-inject顾名思义，也如同上图所示，这里利用image作为RNN隐藏层向量的初始值，即初始`h_state = image`,而对于输入，则如同一般
seq2seq模型，输入为word vectors,输出为word vector后移一个单词，直到预测到<END>标志为止．　　

### Pre-inject  
Pre-inject则将image作为RNN的第一个输入，可以将其视为第一个单词，隐藏层初始状态为随机初始化．　　

### Par-inject  
Par-inject可以理解为pair-inject,及每次输入一个词时需要同时输入image,RNN每次接受两个向量．　　

### Post-inject  
Post-inject则是将image作为最后一个单词输入RNN中．






## Reference  

[^1]: Marc Tanti, Albert Gatt, Kenneth P. Camilleri. Where to put the Image in an Image Caption Generator[J]. 2017.