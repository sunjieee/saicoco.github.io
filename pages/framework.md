---
layout: page
show_meta: false
title: "框架小记"
subheadline: "DL framework使用小记"
permalink: "/framework/"
header:
   image_fullwidth: "dl.jpg"
---
<ul>
    {% for post in site.categories.framework %}
    <li><a href="{{ site.url }}{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
</ul>

