---
layout: post
title: tensorflow使用笔记(1)
tags: [tensorflow]
comments: true
---  

今天开始计划开启一大块，主要关于tensorflow使用的，我想以blog的记录下来。因为实验需要，估计会用tensorflow实现LSTM等复杂操作，以此来熟悉tensorflow。  

还是从一个tutorial开始。一个完整的tensorflow程序可以分为一下几部分：  

* Inputs and Placeholders  
* Build the Graph  
    * Inference  
    * Loss  
* Traing  
* Train the Model  
    * The Graph
    * The Session
    * Train Loop
        * Feed the Graph
        * Check the status
        * Visualize the Status  
        * Save a Checkpoint  
* Evaluate the Model  
* Build the Eval Graph  
* Eval Output  
