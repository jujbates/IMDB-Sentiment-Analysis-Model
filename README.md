# Deploying a Sentiment Analysis Model on SageMaker Project

The project Deploying a Sentiment Analysis by Udacity inc. deploys a recurrent neural network onto AWS for the purpose of determining the sentiment of a movie review using the IMDB data set. This project repo is for educational purpose and should be used as a tool, not for copying. If viewing or using this repo, please follow the Udacity Honor Code and Community Code of Conduct: https://www.udacity.com/legal/community-guidelines

[Introduction](#user-content-introduction)
[Instructions](#user-content-instructions)
[Results](#user-content-results)


# Introduction
The notebook and Python files provided here create a simple web app which interacts with a recurrent neural network once deployed performing sentiment analysis on movie reviews. This project assumes some familiarity with SageMaker, the IMDB Sentiment Analysis using XGBoost and RNNs. 

Before running project here is an image of the finished project, and here are my results.
<img src="final_web_app_deployed_and_reviewing_neg_review.png">

# Instructions
When ready you can run the project by create a AWS Sagemaker notebook instance with this public github repository. This step will involve setting up an AWS IAM role with the AmazonSageMakerFullAccess IAM policy attached. Once created navigate the notebook to the `SageMaker Project.ipynb`.

Once the notebook is open you can simply run the notebook. The notebook will do the following:

1. [Download or otherwise retrieve the data](#user-content-step-1-download-or-otherwise-retrieve-the-data)
2. [Process and Prepare the data](#user-content-step-2-process-and-prepare-the-data)
3. [Upload the processed data to S3](#user-content-step-3-upload-the-processed-data-to-s3)
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.
8. Deploying model on our web app.



Please see the [README](https://github.com/udacity/sagemaker-deployment/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).


## Step 1. Download or otherwise retrieve the data

We will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). This dataset can be found and downloaded here [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

    Maas, Andrew L., et al. Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 2011.


## Step 2. Process and Prepare the data
We will be doing some initial data processing. To begin with, we will read in each of the reviews and combine them into a single input structure. Then, we will split the dataset into a training set and a testing set. Next, we make sure that any html tags that appear in the review are removed. In addition we tokenize our data input, that way words such as entertained and entertaining are considered the same with regard to sentiment analysis. The `review_to_words` method does 5 major things to prep for the vocabulary of a single review. The steps are shown below:

1. Removes all html tags with the line:
```python
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
```
2. Convert all capital letters to lower case letters:
```python
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
```
3. Splits all the words into a list of words
```python
    words = text.split() # Split string into words
```
4. Removes all stopwords, for example the, has, with, I, etc., in English language with the line:
```python
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
```
5. Replaces all English words with the Porter stemming algorithm with the line:
```python
    words = [PorterStemmer().stem(w) for w in words] # stem
```

Once the data is processed we pickle it into `preprocessed_data.pkl`.




## Step 3. Upload the processed data to S3



# Results
<img src="final_web_app_deployed_and_reviewing_neg_review.png">
