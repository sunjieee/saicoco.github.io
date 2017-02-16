---
title: "Mxnet学习笔记(4)--c++源码阅读计划"
layout: post
date: 2017-02-16
tag: mxnet
blog: true
author: karl
category: mxnet
description: 自定义Op
---  

过完年了，新的开始，计划给自己分配点任务:计划将mxnet中的c++源码读一遍，全当学习一遍c++。大致分为以下几个模块
进行阅读：　　

1. operator  
2. optimazer  
3. ndarray  
4. kvstore  
5. executor  

因为在使用python API时感觉没底，最近又遇到自定义op时的打击，于是计划开启学习，目的旨在学习c++和
mxnet中的源码设计，以此来督促自己。其中operator是重点，因为我们自定义层时主要集中在这里。希望可以顺利
进行下去。