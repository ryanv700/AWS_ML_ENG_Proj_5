
# Amazon Distribution Center Inventory Monitoring Project 

## Project Set Up and Installation
This project was completed using Amazon SageMaker Studio within the AWS console. The Image bin dataset was collected by Amazon and is
made available for free at the link here: https://registry.opendata.aws/amazon-bin-imagery/. The resources for this project were collected
and made available by Udacity as part of the AWS Machine Learning nanodegree. The resources to start and run the project are all 
availble in this Github repo.

See the steps below:
1. Create a folder in Sagemaker Studio with the files: Amazon_Bin_Recognition_Completed_Notebook.ipynb, train2.py,
   hpo (1).py and the file_list.json.
2. Run the notebook which will complete the entire end to end project including: Training data download, Split training data into
   training, validation and testing data, Load the data folders to Amazon S3, run a Hyperparameter tuning job to FineTune a ResNet50
   model on the data, and finally create a deployable model with the best hyperparameters incorporating SageMaker profiling and debugging.

The images below serve as a guideline for what you should see in your AWS console as you complete the project:

Training data stored in S3:

<img width="1023" alt="binImages_S3_bucket" src="https://github.com/ryanv700/AWS_ML_ENG_Proj_5/assets/56355045/d74f5035-5176-4d59-ad8b-7708317251fa">

Successful Hyperparameter Tuning Job:

<img width="1111" alt="Successful_Hyperparameter_Tuning_Job" src="https://github.com/ryanv700/AWS_ML_ENG_Proj_5/assets/56355045/4cc5a1c7-e952-4e67-aafa-c18b9be0b236">

Successful model training job:

<img width="1020" alt="Successful_training_job" src="https://github.com/ryanv700/AWS_ML_ENG_Proj_5/assets/56355045/13227129-6732-416c-8486-2aa73943d9db">

Model Artificats Saved in S3:

<img width="1267" alt="model_artifacts_saved_in_S3" src="https://github.com/ryanv700/AWS_ML_ENG_Proj_5/assets/56355045/b5b9a026-7f3a-4e12-9be9-6ee0e3e0103d">


## Dataset

### Overview

Here is a link to the Amazon Bin Dataset https://registry.opendata.aws/amazon-bin-imagery/.

From the website: 
“The Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.”


### Access
The file_list.json is used to download the data from Amazon S3 in a bucket created and maintained by Amazon for the purposes of making this 
dataset available for research.


## Model Training
For the project I used a pretrained ResNet50 model provided by the open source PyTorch Library. The model training, testing, and saving script
train2.py is used to finetune the pretrained model on the image bin dataset and it is used to create a PyTorch Estimator object (which is provided 
by the Sagemaker API) in the notebook. In the notebook the script hpo.py is used to run a Sagemaker Hyperparameter Tuning job. By using AWS 
with a larger compute budget this project can be easily scaled up to a much larger level. To create a model that is production quality for
real world deployment in an Amazon warehouse a more extensive research process using the full 500,000 image dataset which tries several different
models would be necesary.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

