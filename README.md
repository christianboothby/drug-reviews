# **A Hard Pill To Swallow**

## **Understanding how medical drug reviews can help improve the healthcare system**


## Executive Summary

This project was completed during my time on the General Assembly Data Science Immersive course in London.

Through the analysis of my data, which will be web-scraped, I aim to identify review traits which are predictive of positively or negatively rated drugs and use these insights to help provide recommendations for various sectors within the healthcare industry. 

The results of which my analysis was based was achieved through using statistical learning models such as classification predictive modelling, sentiment analysis and natural language processing. The best model that I fitted predicted drug success with 87.6 % accuracy, compared to a baseline of 68.7%. Actionable insights were extracted from this, resulting in the production of a recommendation dataset grouped by condition. Furthermore, analysis into the language used in the reviews was completed in order to identify differences between the positive and negative reviews. Through using the coefficients extracted from the best model, the featured importance of words in the review could actively distinguish between positive and negative reviews. 


## Files in This Repository

 - [**Presentation slides**](): This was prepared to present the project, results, and recommendations to non-technical audience 
-   [**Technical report**](https://github.com/christianboothby/drug-reviews/blob/main/Technical%20Report%20.md): This was prepared for reporting and explaining my project to a technical audience.
-   **Jupyter Notebook files (.ipynb)**:
	-   Step 1: [Dataset and Data Collection](https://github.com/christianboothby/drug-reviews/tree/main/Data%20Collection)
	-   Step 2: [Data Cleaning](https://github.com/christianboothby/drug-reviews/blob/main/Data%20Cleaning.ipynb)
	-   Step 3: [Exploratory Data Analysis](https://github.com/christianboothby/drug-reviews/blob/main/Exploratory%20Data%20Analysis.ipynb)
	-   Step 4: Data Modeling 
		-   [Research Question 1](https://github.com/christianboothby/drug-reviews/blob/main/Research%20Question%201.ipynb)
		-   [Research Question 2](https://github.com/christianboothby/drug-reviews/blob/main/Research%20Question%202.ipynb)

## Problem Statement

Medical drugs are not perfect. It is impossible to design and prescribe a drug which is suitable for each individual person. Doctors and big pharmaceutical companies rely on technical science to make decisions. However, it is possible that the public perception of drugs could be able to shed some new light onto using alternative drugs and make decisions easier to understand for the drug recipient. Through machine learning, could we accurately predict what makes a drug review positive and hence be able to recommend a particular drug for a particular person?

## Objectives

Research objectives are defined as the following:

1.	Obtain drug reviews and review features through web-scraping and clean the reviews
2.	Predict the probability of a drug being rated as positive or negative 
3.	Build a recommendation system for these drugs by condition 
4.	Create a tool that can help doctors, patients and pharmaceuticals improve the likelihood of a positively rated drug

For the purpose of this capstone, the success of a drug is considered to be a positively rated drug, which is a drug rated >= 5. This binary feature has been generated and used as the target variable throughout this capstone. 

## Hypothesis

My hypothesis is that people’s perceptions are significantly predictive of drug success and the language used within the review is important to determining the difference between positive and negative reviews.  

## Research Questions

<a href="https://snipboard.io/6Ylg7p.jpg"><img src="https://snipboard.io/6Ylg7p.jpg" title="source: snipboard.io" style="width: 500px"/></a>

**Question 1 : Can patients’ perceptions allow alternative drugs to be  considered?**

My prediction is that models may not be too accurate in predicting drug success based on the data due to everyone having different definitions of success or failure. However, I think that sentiment analysis into the reviews would reveal some factors into the difference between positivly and negativly rated drugs.

**Question 2: Did more positively rated drugs get different reviews from negatively rated drugs?**

This will further confirm that online patient reviews and perceptions are essential in helping patients make decisions and doctors/pharma understand what patients think about certain drugs. My prediction is that the positive reviews will have shorter review lengths and less helpful value counts. 

## Dataset

I web-scraped data from drugs.com, choosing 30 common conditions and scraping the reviews of the top 10 drugs used to treat each condition. Once the web-scraping ended, I had 58,000 reviews across all drugs. The Pandas DataFrame contained the following variables:
-   Ratings (categorical)
-   Helpful votes (continuous)
-   Reviews
-   Drug 
-   Condition
-   Month
-   Year (2008 – 2020) 

## Exploratory Data Analysis

<a href="https://snipboard.io/AYguFa.jpg"><img src="https://snipboard.io/AYguFa.jpg" title="source: snipboard.io" style="width: 500px"/></a>

## Model Selection
I have set up a binary classification problem (1 as positive and 0 as negative) and so I selected the following classifiers for both research questions 1 and 2:

Note: All the models were run using GridSearchCV.

1.	Logistic Regression 
2.	Random Forest Classifier
3.	Decision Tree Classifier
4.	Supoort Vector Machine 
5.	K-Nearest Neighbours Classifier 

For research question 1, the following additional models were also run:

6.	Ada Boost Classifier
7.	Gradient Boosting Classifier 

In addition to the above, Natural Language Processing (Count Vectorization) was used throughout the modelling. 

## Findings

### Question 1. Can patients’ perceptions allow alternative drugs to be  considered? What features of the review are important?

Text features succeeded in having significant indications for the success of the drug, allowing a ranking of drugs per condition to be formed. Logisitic Regression model with ridge penalty predicted drug success with 88% accuracy. A visual representation of the models I fitted, with the test and CV scores can ben seen below:

<a href="https://snipboard.io/paRi5H.jpg"><img src="https://snipboard.io/paRi5H.jpg" title="source: snipboard.io" style="width: 500px"/></a>

I decided to use a Harvard dataset containing a list of 1915 positive words and 2291 negative words and undertook a series of steps of feature engineering to create a Pandas DataFrame of ranked drugs based on positive predictions. 

[Harvard Dataset](http://www.wjh.harvard.edu/~inquirer/homecat.htm)

**Note: Only first three conditions, ranked alphabetically, have been shown for the purpose of example**

<a href="https://snipboard.io/zpcEPl.jpg"><img src="https://snipboard.io/zpcEPl.jpg" title="source: snipboard.io" style="width: 500px"/></a>

### Question 2. Did more positively rated drugs get different reviews from negatively rated drugs? What differentiates positive from negative?

The word count between the positive and negative reviews did not differ. Therefore, this doesn’t give us much information, so I went on to look at the coefficients provided for the best model fitted from Q1. This proved that the positive and negative reviews have very different language as expected.

**Positive Reviews:**

<a href="https://snipboard.io/rS6Yiy.jpg"><img src="https://snipboard.io/rS6Yiy.jpg" title="source: snipboard.io" style="width: 500px"/></a>

**Negative Reviews:**

<a href="https://snipboard.io/8admkx.jpg"><img src="https://snipboard.io/8admkx.jpg" title="source: snipboard.io" style="width: 500px"/></a>

The words side effects are most common in the plotted bigrams and so I used a list of common side-effects mentioned in the reviews to predict drug sucess but this did not give scores above the baseline (0.687).

Despite this, I still wanted to understand what features of the review were impactful on the rating. Therefore, I decided to fit one more set of models on some new predictor variables. I used feature engineering to create some new variables. These were word count, sentence count, unique word count, letter count, punctuation count, stop word count, mean word length and sentiment which was caculated using the polarity of the reviews. I used these new features combined with the helpful count, month, year, condition and drug as predictors. Having identified the Random Forest Classifier as the best model, and running it through a grid search, the best score was calculated as being 0.787. I then took this model and identified the feature importance’s of the predictors to identify the most important predictors in reaching this score. A graphical representation of this has been shown below: 

<a href="https://snipboard.io/QVOhKC.jpg"><img src="https://snipboard.io/QVOhKC.jpg" title="source: snipboard.io" style="width: 500px"/></a>


## Conclusion

### To Non-Technical Audience 

Through my research steps, it was discovered that:
- The performance of text features was better than anticipated – patient perceptions extracted from reviews were useful in predicting positive drug ratings
- The performance of text features predicted positive drug ratings with 88% accuracy
- This led to forming recommendations based on our predictions, the ratio of positive words per review and the helpful vote count. 
- Positive and negative reviews has similar word counts
- Extracting model coefficients proved that different language was used for different rating category. 
- The mention of side-effects in the review did not help predict drug success
- The performance of engineered review features allowed predictions of drug success to be made with 78% accuracy
- Sentiment of the review was the most important feature in this prediction 
- This leads to the conclusion of putting more focus on patient perception of drugs to prescribe, invest or synthesize drugs. 


### To Technical Audience
The goal of this project was to look for features within public written reviews to better predict the success of medical drugs. I generated features through several aspects and investigated the use of text features as predictors. 

The model was successful in predicting positively rated drugs with a score of 87.6% (vs a 68.7% baseline). This was a grid search with logistic regression model and ridge penalty, using the review text with an ngrams range of (1,2). 

The investigation into the mention of side effects in the reviews failed to produce any models with accuracy scores higher than the baseline. Hence, it was concluded that side effects do not impact the prediction of positive drugs. 

Finally, a grid search with random forest classifier identified that the review sentiment was most important in predicting positively rated drugs, performing with 78% accuracy. 

This leads to the conclusion of putting more focus on patient perception of drugs to prescribe, invest or synthesize drugs


