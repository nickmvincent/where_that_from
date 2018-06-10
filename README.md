# What currently lives here?
This code was for my EECS349 final project (described below).
The code is part of a larger personal project with the goal of automatically detecting sentences that have quantitative dependencies (i.e. code and data) and helping to link text and code together. The larger project is currently on hold.


About the EECS349 project (for the full report, open the "report" Jupyter notebook or visit http://www.nickmvincent.com/eecs349_report.html): 

# Citation Needed? Automatically Detecting Sentences from Computing Research Papers that Need Citations
Author: Nicholas Vincent, Northwestern University

email: nickvincent@u.northwestern.edu | web: [www.nickmvincent.com](http://www.nickmvincent.com)

Prof. Doug Downey's EECS349 course, final project

Code for this project: https://github.com/nickmvincent/where_that_from

## Abstract: 
The scientific paper is the primary artifact produced by scientists of every discipline. Within a scientific paper, citations are incredibly valuable: they help connect a paper to the surrounding literature, provide evidence for claims, and empower a single PDF (often with less than 10 pages) to “stand on the shoulders of giants” (as Isaac Newton put it) by referencing prior work. However, language is ambiguous, and it may not always be trivial to decide whether a certain sentence should include a citation or not.  Therefore, it may be valuable for an author or reviewer to be able to quickly identify whether a sentence should include a citation or not, or better yet to get a list of sentences that are worthy of deeper inspection. We implement and test a variety of machine learning classifiers that attempt to solve the task of identifying whether or not a given sentence should include a citation.

In this document, we report our results for a variety of approaches, falling into two broad categories: (1) using word count vectorization alongside textual metadata with traditional classification techniques (Naive Bayes, Decision Trees, Logistic Regression, etc.) and (2) using deep learning techniques. Using a dataset with 6022 examples (14.8% positive), we see that logistic regression and support vector machine approaches provide a good balance of performance and quick training. Finally, using a larger dataset with 32,228 examples (12.4% positive), we are able to train a classifer with 75% recall and 38.5% precision (AUROC is 87.8% and accuracy is 89%). This performance should be adequate to make this classifier useful as a "machine assistant" for authors or reviewers of academic papers.