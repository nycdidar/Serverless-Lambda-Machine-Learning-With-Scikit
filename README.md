# Serverless(Lambda) Machine Learning With Scikit
Serverless(Lambda) Machine Learning With Scikit

##### Installation steps
* Go to https://console.aws.amazon.com/iam/home?region=us-east-1#/security_credentials
* Create `New Access Key`
* Clone this repo.
* `cd Serverless(Lambda) Machine Learning With Scikit`
* RUN `serverless config credentials --provider aws --key YOUR_KEY --secret YOUR_SECRET`
* RUN ` sls plugin install -n serverless-python-requirements` 

##### Run and deploy
* Train model `python train.py`
* Deploy app `sls deploy -v`
* Test prediction `curl 'https://wbzhw5yw07.execute-api.us-east-1.amazonaws.com/dev/?' --data 'input_data=5.9,3.0,5.1,1.8'`
* To destroy and remove lambda/api `sls remove`.

