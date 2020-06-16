import mlflow.sagemaker as mfs

# AWS env and image url setup
arn = "arn:aws:iam::197306934454:role/service-role/AmazonSageMaker-ExecutionRole-20200429T114488"
region = 'us-east-1'
app_name = "mlflow-sagemaker-demo" # Name of the app that will be deployed
model_uri = "mlruns/2/6d4a49d611f1416fb4adc115ea7abd04/artifacts/direct-marketing-xgboost-model"
image_url = "197306934454.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:1.8.0" 

# Deploy the model to sagemaker
mfs.deploy(app_name = app_name,
           model_uri = model_uri,
           region_name = region,
           mode = "replace", 
           execution_role_arn = arn,
           image_url = image_url)

