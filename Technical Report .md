# **Technical Report**

# **A Hard Pill To Swallow**

## **Understanding how medical drug reviews can help improve the healthcare system**

## Overview

This project was completed during my time on the General Assembly Data Science Immersive course in London.

Through the analysis of my data, which will be web-scraped, I aim to identify review traits which are predictive of positively or negatively rated medical drugs and use these insights to help provide recommendations for various sectors within the healthcare industry. 

The results of which my analysis was based was achieved through using statistical learning models such as classification predictive modelling, sentiment analysis and natural language processing. The best model that I fitted performed with 87.6 % accuracy, compared to a baseline of 68.7%. Actionable insights were extracted from this, resulting in the construction of a recommendation dataset of positivly ranked drugs grouped by condition. Furthermore, analysis into the language used in the reviews was completed in order to identify differences between the positive and negative reviews. Through using the coefficients extracted from the best model, the featured importance of words in the review could actively distinguish between positive and negative reviews. 

## Motivation

I have always been fascinated with the healthcare industry and the complex network of sectors behind it. Having transitioned from research into data science, I knew I wanted to investigate how data science and the healthcare industry are intertwined. This brought me to this project, looking at how the public perception of drugs can be used to make insights into drug success or failure. Having come from research, working on synthesising complex drug structures, I am aware of how pharmaceutical companies, and doctors alike, do not take the public’s perception of a drug into consideration. Is there more to understand and apply beyond the theoretical science and can machine learning help uncover this?

## Problem Statement

Medical drugs are not perfect. It is impossible to design and prescribe a drug which is suitable for each individual person. Doctors and big pharmaceutical companies rely on technical science to make decisions. However, is it possible that the public perception of drugs could be able to shed some new light onto using alternative drugs and make decisions easier to understand for the drug recipient. Through machine learning, could I accurately predict what makes a drug review positive and hence be able to recommend a particular drug for a particular person?

## Objectives

Research objectives are defined as the following:

1.	Obtain drug reviews and review features through web-scraping and clean the reviews
2.	Predict the probability of a drug being rated as positive or negative 
3.	Build a recommendation system for these drugs by condition 
4.	Create a tool that can help doctors, patients and pharmaceuticals improve the likelihood of a positively rated drug

For the purpose of this capstone, a positively rated drug is a drug rated >= 5. This binary feature has been generated and used as the target variable throughout this capstone. 

## Hypothesis

My hypothesis is that people’s perceptions are significantly predictive of drug success and the language used within the review is important to determining the difference between positive and negative reviews.  

## Research Questions

<a href="https://snipboard.io/6Ylg7p.jpg"><img src="https://snipboard.io/6Ylg7p.jpg" title="source: snipboard.io" style="width: 500px"/></a>

**Question 1 : Can patients’ perceptions allow alternative drugs to be considered?**

My prediction is that models will be accurate in predicting drug success based on the data. I also predict that sentiment analysis into the reviews would reveal some factors into the difference between positivly and negativly rated drugs.

**Question 2: Did positively rated drugs get different reviews from negatively rated drugs?**

This will further confirm that online patient reviews and perceptions are essential in helping patients make decisions and doctors/pharma understand what patients think about certain drugs. My prediction is that the positive reviews will use more uplifting language than the negative reviews. 

## Research Steps

**Part 1: Pitch & Problem Statement**
Define the problem statement, potential audience, goals, success metrics and data sources.

**Part 2: Dataset and Data Collection**
Source and format the required data for the project, perform preliminary data munging and cleaning of the data. 

**Part 3: EDA and Preliminary Analysis**
Quantitatively describe and visualize the data.

**Part 4: Data Modelling**
Design models that accurately predict drug success. Identify key features that increase the accuracy of the model. Evalutate model performance and discuss results. 

**Part 5: Presentation**
Complete a 20 minute presentation of my project for a non-technical audience. Discuss goals, success criteria, data, approach, model, findings, limitations and future implications. 

## Dataset

I web-scraped data from drugs.com, choosing 30 common conditions and scraping the reviews of the top 10 drugs used to treat each condition. Once the web-scraping ended, I had 58,000 reviews across all drugs. The Pandas DataFrame contained the following variables:
-   Ratings (categorical)
-   Helpful votes (continuous)
-   Reviews
-   Drug 
-   Condition
-   Month
-   Year (2008 – 2020) 

<a href="https://snipboard.io/3aA5qn.jpg"><img src="https://snipboard.io/3aA5qn.jpg" title="source: snipboard.io" style="width: 500px"/></a>

**At this stage, the DataFrame has 58,0000 data entries and 7 columns**

## Data Cleaning and Feature Engineering 

I defined the data cleaning process as the following:

**Step 1:** Load the web-scraped data and concatenate to one Pandas DataFrame 

Each condition and their subsequent drugs and reviews were scraped separately, creating 30 separate csv files. Therefore, first I had to concatenate all of the condition reviews into one Pandas DataFrame, resetting the index, dropping duplicates and rows containing NaN values as I did so.

**Step 2:** Clean variables 

I transformed the date column into year and month by using the split function and assigning new column variables. I then dropped the original date variable. Using regex, I cleaned the review column. I removed the text jargon and transformed it to lower case by applying functions to the variable. I imported a dictionary of contractions and applied these to the review text. Finally, I also used English stop words and a manual list of punctuation to remove further unnecessary words and symbols.

Further data cleaning and feature engineering was completed throughout the project, which I will explain at the point of relevance. 

**At this stage, the DataFrame has 58,0000 data entries and 9 columns**

## Exploratory Data Analysis 

<a href="https://snipboard.io/AYguFa.jpg"><img src="https://snipboard.io/AYguFa.jpg" title="source: snipboard.io" style="width: 500px"/></a>

The review count per year indicates that there has been a general increase from 2008 -2019 in the number of reviews written. It is worth noting that 2020 has considerably less because I scraped the data in October of that year so the year was not completed. After looking at the distribution of the months in the year in accordance with the review count, I decided to drop the 2020 year.

People are more likely to rate higher than lower according to the count per rating histogram. Likewise, people are more likely to vote at extremes (1 or 10) rather than middle of the road. This shows that there will be a class imbalance in our target variable, which needs to be considered in the modelling and analysis. 

The most reviewed conditions seem to surround mental health more than physical health.  

The average helpful count seems to be increasing a lot between having a rating of 6 and 10. This means people are more likely to vote for a helpful review if it is more highly rated. This could be a potential for a good predictor in our model. The Average review length peaks at 8 although it does still increase overall between 1 and 10. This could be another possible predictor in our modelling. 

**At this stage, the DataFrame has 54,0000 data entries and 9 columns**

## Model Selection 

I have set up a binary classification problem (1 as positive and 0 as negative) and so I selected the following classifiers for both research questions 1 and 2:

1.	Logistic Regression 
2.	Random Forest Classifier 
3.	Decision Tree Classifier 
4.	Support Vector Machine 
5.	K-Nearest Neighbours Classifier

In an effort to improve accuracy I used ensemble learning algorithms, which are built combining weak uncorrelated models increasing the weights/importance of misclassified observations on each iteration to obtain better predictions. I started with an Ada Boost Classifier which is known to robust a strong classifier by ensuring the accurate predictions of unusual observations and minimizing training error. Followed by a Gradient Boosting classifier which differs by fitting the new predictor to the residual errors made by the previous predictor. Therefore, for research question 1, the following additional models were also fitted:

6.	Ada Boost Classifier
7.	Gradient Boosting Classifier 

In addition to the above, Natural Language Processing (Count Vectorization) was used throughout the modelling. 

**Note: all the models were run using GridSearchCV**

The models were evaluated using:
- Baseline: The minimum score that would be achieved without any models run (or with ill-fitted models)
- Accuracy Test Score: Represents how well the model can perform on unseen data
- Mean Cross-Validation Score (CV Score): Measures the consistency of the model

The success metrics used to analyse the models were:
- Accuracy Test Score
- Precision Score 
- Recall Score 
- Confusion Matrix
- Classification Report 
- Precision – recall curve 
- ROC Curve 

## Findings

**Summary**

Text features succeeded in having significant indications for a positively rated drugs, allowing a ranking of drugs per condition to be formed. Classification models predicted drug success with 88% accuracy. Additionally, extracting the coefficients of this model gave the word feature importance’s for both rankings of reviews. Moreover, using engineered review features as predictors saw classification models predict drug success with 79% accuracy. Despite this, the presence of side effects in the reviews did not have any significant indications for drug success. 

Baseline = 0.687 

**Question 1: Can patients’ perceptions allow alternative drugs to be considered?**

In order to get an overview of which predictor features to use, I began with running 5 default logistic regression models using 5 different sets of predictors. A summary table of models, predictors, test scores and CV scores has been provided below:

<a href="https://snipboard.io/dDn84s.jpg"><img src="https://snipboard.io/dDn84s.jpg" title="source: snipboard.io" style="width: 500px"/></a>

The models with predictors not using the review text itself had considerably lower scores than when the text was included, via a count vectorizer classifier. I used a pipeline to initiate the review-based models to ensure vectorization occurred. Logistic Regression with an ngrams range between 1 and 2 gave the best test score with 0.876. This is a combination of single words and consecutive 2 words together. The difference in scores between models has been visualised in the horizonatal bar plot below: 

<a href="https://snipboard.io/QvMLub.jpg"><img src="https://snipboard.io/QvMLub.jpg" title="source: snipboard.io" style="width: 500px"/></a>

Following the results of this initial modelling process, I continued with the top scoring Logistic Regression (ngrams(1,2)) model and refined the model to achieve better scores. In addition, I also ran further models using the ngrams(1,2) predictor to investigate whether other models could achieve better scores. All of these models were run in a pipeline in order to use the Count Vectorizer on the text. The difference in scores between models has been visualised in the horizonatal bar plot below:

**Note: * indicates the models were run using GridSearchCV**

<a href="https://snipboard.io/NEwxcz.jpg"><img src="https://snipboard.io/NEwxcz.jpg" title="source: snipboard.io" style="width: 500px"/></a>

Despite running grid searches on the majority of the models with a wide range of parameters, the best score and hence model remained as Logistic Regression (penalty = ‘l2’, C =. 0.35938, solver = ‘liblinear’, fit_intercept =False, max_iter = 1000). The accuracy score remained at 0.876. This has been visualised below:

<a href="https://snipboard.io/paRi5H.jpg"><img src="https://snipboard.io/paRi5H.jpg" title="source: snipboard.io" style="width: 500px"/></a>

Success Metrics: 

**On test set: 10,923 data entries**

<a href="https://snipboard.io/vSLPEI.jpg"><img src="https://snipboard.io/vSLPEI.jpg" title="source: snipboard.io" style="width: 500px"/></a>

Overall the model performed very well.

The classification report shows that the precison, recall and f1-scores scores match the accuarcy scores provided by our model. 

The confusion matrix gives the following statstics:
- True Positives = 64%
- True Negatives = 24%
- False Positives = 5%
- False Negatives = 7%

The precsion-recall curve identified that class 1 (positive ratings) are more likely to be predicted correctly that class 0 (negative ratings). This was identified by calculating the area under the curve, which was a ratio of 0.959:0.863 for class 1 and 0 respectivly. This slight deviation in areas between the curves is expected due to class 1 being the majority class. Although class 0 was lower, these values are both very high and reflect our model performed well. 

The ROC curve also provided evidence that the model performed well, with both class 0 and 1 having high areas under the curve with 0.93. 

Following this, I continued in gathering insights from this model and interpreting the results in terms of the success of predicting positive drug ratings. 

I decided to use a Harvard dataset containing a list of 1915 positive words and 2291 negative words and undertook a series of steps of feature engineering to create a Pandas DataFrame of ranked drugs based on positive predictions. 

[Harvard Dataset](http://www.wjh.harvard.edu/~inquirer/homecat.htm)

The feature engineering steps and construction of the resulting DataFrame are summarised below:

1. Count number of positive and negative words in each review according to Harvard dataset
2. Calculate positive word ratio and transform to 1,0.5,0 for positive, undecided or negative respectively 
3. Group by condition and calculate number of reviews per condition (denoted condition size)
4. Scale helpful votes:
    Helpful Scaled=  Helpful Count/Condition Size
5. Calculate rating predictions based on best model (Logistic Regression(ngrams(1,2)) with Ridge penalty 
6. Calculate final predictions:
    Final Prediction=(Model Predictions+Postive Ratio) × Helpful Scaled
7. Group by condition and aggregate by final prediction mean for each drug

**Only the first three conditions, ranked alphabetically, have been shown for example purposes**

<a href="https://snipboard.io/zpcEPl.jpg"><img src="https://snipboard.io/zpcEPl.jpg" title="source: snipboard.io" style="width: 500px"/></a>

This table gives a final ranking of drugs, grouped by condition, based on three factors:
1. Model Predictions
2. Postive word ratio 
3. Helpful vote count 

Overall, the higher the total mean prediction, the more positivly rated the drug according to our model and our dataset. This could be used to look for alternative drugs on demand and successfully answers research question 1. 

**Question 2: Did positively rated drugs get different reviews from negatively rated drugs?**

In order to continue the analysis into the use of drug reviews in predicting drug ratings, I looked into the relationship between positive and negative reviews. When investigating the word count between the positive and negative reviews, I produced word clouds, unigrams and bigrams. These have been visualised below:

<a href="https://snipboard.io/mXRPY7.jpg"><img src="https://snipboard.io/mXRPY7.jpg" title="source: snipboard.io" style="width: 500px"/></a>

From the visualisations, it is clear to see that there is not much difference between the word counts in positive and negative reviews. From the bigrams, it is clear that 'side effects' is extremely common in both positive and negative reviews. Therefore, I decided to investigate this further and create a list of common side effects. I then used this list to place a binary column for each review if the side effect was mentioned. 

The common side effects used were diarrhea, constipation, dizziness, drowsiness, fatigue, palpitations, nausea, vomitting, rash, upset stomach, hives, headache, weight gain, weight loss, dry, suicidal, fever, swelling, alopecia, heartburn, burning, dryness, vomitting, pain, anxiety, suicide and mood swings.

After fitting the same models as Q1, it was clear that models didn't provide increased accuracy. This was found due to the test and mean cross validation scores not exceeding the baseline score (0.687). Therefore, it was concluded that the mention of side effects in the review failed to predict positive drug ratings. A summary of the results from these models can be found below:

<a href="https://snipboard.io/eG6DYX.jpg"><img src="https://snipboard.io/eG6DYX.jpg" title="source: snipboard.io" style="width: 500px"/></a>

After this, I reverted back to the best predictor (ngrams(1,2)) and drew some more insights. I went on to investigate the difference in language betweeen the positive and negative reviews, which was done by extracting the coefficients of the best model from research question 1 (Logistic Regression with Ridge penalty). I was able to sort the coefficients for each word or two words and ultimatly form a table with the most important words in positive and negative words, ranked by positive impact. This can be found below: 

<a href="https://snipboard.io/fwpXML.jpg"><img src="https://snipboard.io/fwpXML.jpg" title="source: snipboard.io" style="width: 500px"/></a>

Finally, in order to investigate all aspects of the drug reviews, I decided to undertake some further feature engineering and use some review features as predictors to predict positive ratings. The new variables that I engineered were review sentiment, mean word length, stop word count, letter count, unique word count, punctuation count, general word count, sentence count and title word count. I ran 5 models with a gridsearch using these new review features, helpful count, condition, month and year as predictor variables. A summary table of models, test scores and CV scores has been provided below along with the visualisation of these scores in a horizontal bar plot:

<a href="https://snipboard.io/lmcBe5.jpg"><img src="https://snipboard.io/lmcBe5.jpg" title="source: snipboard.io" style="width: 500px"/></a>

<a href="https://snipboard.io/anwPyM.jpg"><img src="https://snipboard.io/anwPyM.jpg" title="source: snipboard.io" style="width: 500px"/></a>

The model which performed the best in terms of highest test and mean cross validation score was the Random Forest Classifier, with an accuracy score of 0.787 (compared to a baseline of 0.687). Although this score is not as high as the model fitted in research question 1, it is still high enough to conduct some analysis. I extracted the feature importaces according to the Random Forest Classifier and plotted them, ranking from highest to lowest. This visualisation can be seen below: 

<a href="https://snipboard.io/QVOhKC.jpg"><img src="https://snipboard.io/QVOhKC.jpg" title="source: snipboard.io" style="width: 500px"/></a>

From this plot, the feature that has the most importance is the sentiment of the review, calculated using the polarity function applied on the review text. This again confirms that public perception is vital to understanding the drug market within the healthcare industry. 

## Conclusion

### Non-Technical Audience 

Through my research steps, it was discovered that:
- The performance of text features fulfilled by hypothesis that they are important in predicting positive drug ratings
- The performance of text features predicted positive drug ratings with 88% accuracy
- This led to forming recommendations based on our predictions, the ratio of positive words per review and the helpful vote count
- Positive and negative reviews has similar word counts for the most frequently used words
- Extracting model coefficients proved that different language was used for different rating category 
- The mention of side-effects in the review did not help predict drug success
- The performance of engineered review features allowed predictions of positive ratings to be made with 78% accuracy
- Sentiment of the review was the most important feature in this prediction 
- This leads to the conclusion of putting more focus on patient perception of drugs to prescribe, invest or synthesize drugs


### Technical Audience

The goal of this project was to look for features within public written reviews to better predict postively rated drugs. I generated features through several aspects and investigated the use of text features as predictors. 

The model was successful in predicting positively rated drugs with a score of 87.6% (vs a 68.7% baseline). This was a grid search with logistic regression model and ridge penalty, using the review text with an ngrams range of (1,2) as a predictor. 

The investigation into the mention of side effects in the reviews failed to produce any models with accuracy scores higher than the baseline. Hence, it was concluded that side effects do not impact the prediction of positive drugs. 

Finally, a grid search with random forest classifier identified that the review sentiment was most important in predicting positively rated drugs, performing with 78% accuracy. 

This leads to the conclusion of putting more focus on patient perception of drugs to prescribe, invest or synthesize drugs.

## Limitations

Due to the limited time frame, I only had time to web-scrape one website and had to be selective with the conditions and number of drugs per condition that I chose. Thus, the dataset may not be an accurate representation of the drug market. Moreover, the website did not give any information about location, which could be potentially important considering different countries have different healthcare systems in place such as insurance or private and public options. This could potentially cause low reliability of reviews as someone who pays for a drug and has a negative experience is more likely to rate lower than if the drug is claimed on insurance. 

Moreover, it is a known fact that sentiment analysis has low reliability when the number of positive and negative words is small. For example, if there are 0 positive words and 1 negative word according to the Harvard dataset, the review is classified as negative. Therefore, a limit on sentiment words would be necessary to ensure higher reliability. To further ensure reliability of the predicted values, I normalized the helpful votes count and multiplied it by the predicted values. However, useful count will be higher for older reviews as the number of cumulated site visitors increases. This insinuates that I would need to consider time into the normalisation of the helpful vote counts.

## Future Recommendation 

For future research, I would recommend to:

- Obtain more information through web scraping, including alternative review websites
- Include more columns such as:
    o	Dosage
    o	Interactions with other drugs 
    o	Known side effects 
- Fit more models and run more grid searches
- Build recommendation engines for doctors to use for their patients 
- Build recommendation engines for pharmaceutical companies to use in their business models
- Deploy model on AWS to give more computing power and a better choice of models





