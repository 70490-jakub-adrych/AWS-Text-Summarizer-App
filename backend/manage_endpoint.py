import boto3
import argparse
import time
import json

def get_endpoint_status(endpoint_name):
    """Get the current status of a SageMaker endpoint"""
    client = boto3.client('sagemaker')
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            return "DELETED"
        else:
            raise e

def check_model_exists(model_name):
    """Check if a SageMaker model exists"""
    client = boto3.client('sagemaker')
    try:
        client.describe_model(ModelName=model_name)
        return True
    except client.exceptions.ClientError:
        return False

def deploy_model(model_name, model_path=None):
    """Deploy a pre-trained Hugging Face model to SageMaker"""
    try:
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel
    except ImportError:
        print("Error: sagemaker package is required for deployment. Install it with:")
        print("pip install sagemaker")
        return False

    print(f"Deploying model '{model_name}'...")
    
    try:
        # Get the execution role
        role = sagemaker.get_execution_role()
    except ValueError:
        # If running locally, we'll need to specify the role ARN
        print("Running outside of SageMaker, please specify your SageMaker role ARN:")
        role = input("Role ARN: ")

    # HuggingFace model configuration
    hub = {
        'HF_MODEL_ID': 'sshleifer/distilbart-cnn-6-6',  # Default model for summarization
        'HF_TASK': 'summarization'
    }
    
    try:
        huggingface_model = HuggingFaceModel(
            transformers_version='4.26',
            pytorch_version='1.13',
            py_version='py39',
            env=hub,
            role=role,
            model_name=model_name  # Use the specified model name
        )
        
        # Create the model without creating an endpoint yet
        huggingface_model.create_model()
        print(f"Model '{model_name}' created successfully")
        return True
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        return False

def stop_endpoint(endpoint_name):
    """Delete a SageMaker endpoint (stopping billing)"""
    client = boto3.client('sagemaker')
    
    status = get_endpoint_status(endpoint_name)
    
    if status == "DELETED":
        print(f"Endpoint '{endpoint_name}' is already deleted.")
        return
    
    print(f"WARNING: This will DELETE the endpoint '{endpoint_name}', not just pause it.")
    print("You will need to start it again later, which may take a few minutes.")
    confirm = input("Do you want to proceed? (y/N): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    print(f"Deleting endpoint '{endpoint_name}'...")
    client.delete_endpoint(EndpointName=endpoint_name)
    
    # Wait for endpoint to be fully deleted
    while True:
        try:
            status = get_endpoint_status(endpoint_name)
            print(f"Current status: {status}")
            time.sleep(10)
        except:
            print(f"Endpoint '{endpoint_name}' has been successfully deleted.")
            break

def start_endpoint(endpoint_name, model_name):
    """Start a SageMaker endpoint from an existing model"""
    client = boto3.client('sagemaker')
    
    try:
        status = get_endpoint_status(endpoint_name)
        if status != "DELETED":
            print(f"Endpoint '{endpoint_name}' already exists with status: {status}")
            return
    except:
        pass  # Endpoint doesn't exist, which is what we want
    
    # Check if model exists
    model_exists = check_model_exists(model_name)
    if not model_exists:
        print(f"Model '{model_name}' not found. Attempting to deploy it now...")
        if not deploy_model(model_name):
            print("Model deployment failed. Please run deploy_pretrained_model.py first.")
            return
    
    # Create endpoint configuration
    config_name = f"{endpoint_name}-config"
    print(f"Creating endpoint configuration '{config_name}'...")
    
    try:
        client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.m5.xlarge',  # Use the same instance type as before
                    'InitialInstanceCount': 1
                }
            ]
        )
    except client.exceptions.ClientError as e:
        if "already exists" not in str(e):
            raise e
    
    # Create endpoint
    print(f"Creating endpoint '{endpoint_name}'...")
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    
    # Wait for endpoint to be ready
    print("Waiting for endpoint to be ready (this may take a few minutes)...")
    while True:
        status = get_endpoint_status(endpoint_name)
        print(f"Current status: {status}")
        
        if status == "InService":
            print(f"Endpoint '{endpoint_name}' is now running and ready to use!")
            break
        elif status in ["Failed", "OutOfService"]:
            print(f"Endpoint creation failed with status: {status}")
            break
            
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage SageMaker endpoints')
    parser.add_argument('action', choices=['start', 'stop', 'status'], 
                      help='Action to perform (start=create/start endpoint, stop=delete endpoint, status=check status)')
    parser.add_argument('--endpoint-name', default='huggingface-pytorch-inference-2025-05-08-18-46-16-895', help='Name of the SageMaker endpoint')
    parser.add_argument('--model-name', default='huggingface-pytorch-inference-2025-05-08-18-46-16-895', help='Name of the SageMaker model (needed for start)')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        start_endpoint(args.endpoint_name, args.model_name)
    elif args.action == 'stop':
        stop_endpoint(args.endpoint_name)
    elif args.action == 'status':
        status = get_endpoint_status(args.endpoint_name)
        print(f"Endpoint '{args.endpoint_name}' status: {status}")
        
        if status == "DELETED":
            model_exists = check_model_exists(args.model_name)
            if model_exists:
                print(f"Note: The model '{args.model_name}' exists, so you can start the endpoint again.")
            else:
                print(f"Note: The model '{args.model_name}' does not exist. You will need to deploy it first.")
