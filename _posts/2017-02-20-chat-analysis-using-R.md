---
layout: post
title: Chat analysis
---

# Chat analysis (google hangout and whatsapp)
little experiment I did last year, using R NLP librarires and hangout and whatsapp data.

### [**GitHub Link**](https://github.com/ranuzz/chatAnalysis) 

### Getting/Cleaning hangout data
Hangout data can be downloaded using google takeaway. The data that I used and tested is from 2014-2015 timeframe.
JSON file provided by google is very huge and requires knowledge of data format to 
extract the conversation so I used https://bitbucket.org/dotcs/hangouts-log-reader/ tool to extract the convertion

after processing the output of 'hangout_reader.py' I had the file in this format:

> 2014-02-18 16:44:22: \<1author-1> message <br>
> 2014-02-18 16:44:33: \<2author-2> message

### Getting/Cleaning whatsapp data
Getting whatsapp data was straight forward. Just emailed the conversation to my inbox. only problem was that the format
slightyly changed in between 2014 and 2015. so I had to use two different parsing code.

2014 format
> 09:53, 21 Jan - 1author-1: message <br>
> 09:57, 24 Jan - 2author-2: message
 
2015 format
> 20/05/2015, 1:18 PM - 1author-1: message <br>
> 20/05/2015, 1:18 PM - 2author-2: message

### Final data format
Final data was exported to individual CSV files *(cleanChat.c)* with following columns
* year
* month
* day
* hour
* minute
* author
* message
* source

### Data input
datain.R : to input data

### WordCloud
wordcloud.R : Generates a basic wordcloud
libraries
* slam
* tm
* wordcloud

### GGplot2
ggplot2.R : Generates a basic graph
library
* ggplot2

### Other details
R version : 3.2.5

### further work
will add more experiments soon.
