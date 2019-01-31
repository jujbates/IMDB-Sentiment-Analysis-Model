# Deploying a Sentiment Analysis Model on SageMaker Project

The project Deploying a Sentiment Analysis by Udacity inc. deploys a recurrent neural network onto AWS for the purpose of determining the sentiment of a movie review using the IMDB data set. This project repo is for educational purpose and should be used as a tool, not for copying. If viewing or using this repo, please follow the Udacity Honor Code and Community Code of Conduct: https://www.udacity.com/legal/community-guidelines

The notebook and Python files provided here create a simple web app which interacts with a recurrent neural network once deployed performing sentiment analysis on movie reviews. This project assumes some familiarity with SageMaker, the IMDB Sentiment Analysis using XGBoost and RNNs. 

Before running project here is an image of the finished project, and here are my results.
<img src="final_web_app_deployed_and_reviewing_neg_review.png">

When ready you can run the project by create a AWS Sagemaker notebook instance with this public github repository. This step will involve setting up an AWS IAM role with the AmazonSageMakerFullAccess IAM policy attached. Once created navigate the notebook to the `SageMaker Project.ipynb`.

Once the notebook is open you can simply run the notebook. The notebook will do the following:

0. [Download or otherwise retrieve the data](#user-content-step-1-download-or-otherwise-retrieve-the-data)
1. Process / Prepare the data.
2. Upload the processed data to S3.
3. Train a chosen model.
4. Test the trained model (typically using a batch transform job).
5. Deploy the trained model.
6. Use the deployed model.
7. Deploying model on our web app.



Please see the [README](https://github.com/udacity/sagemaker-deployment/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).


## Step 1. Download or otherwise retrieve the data
