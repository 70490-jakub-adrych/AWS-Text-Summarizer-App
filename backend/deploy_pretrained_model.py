import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import argparse

def deploy_model(instance_type='ml.m5.xlarge', endpoint_name='summarizer-endpoint'):
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Display some information
    print(f"SageMaker role: {role}")
    print(f"SageMaker session bucket: {session.default_bucket()}")
    
    # Configure the Hugging Face model
    hub = {
        'HF_MODEL_ID': 'facebook/bart-large-cnn', # model_id from hf.co/models
        'HF_TASK': 'summarization'                # NLP task you want to use
    }

    # Create HuggingFaceModel
    huggingface_model = HuggingFaceModel(
        env=hub,
        role=role,
        transformers_version='4.17.0',
        pytorch_version='1.10.2',
        py_version='py38',
        entry_point='inference.py',  # The script we just created
    )
    
    # Deploy the model
    print(f"Deploying model to endpoint: {endpoint_name}")
    print(f"Using instance type: {instance_type}")
    
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print(f"Model deployed successfully to endpoint: {endpoint_name}")
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge",
                        help="Instance type for deployment")
    parser.add_argument("--endpoint-name", type=str, default="summarizer-endpoint",
                        help="Name of the SageMaker endpoint")
    
    args = parser.parse_args()
    deploy_model(args.instance_type, args.endpoint_name)
