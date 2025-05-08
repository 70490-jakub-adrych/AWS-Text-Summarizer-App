import boto3
import argparse
import time

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

def stop_endpoint(endpoint_name):
    """Stop/delete a SageMaker endpoint"""
    client = boto3.client('sagemaker')
    
    status = get_endpoint_status(endpoint_name)
    
    if status == "DELETED":
        print(f"Endpoint '{endpoint_name}' is already deleted.")
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
    try:
        client.describe_model(ModelName=model_name)
    except client.exceptions.ClientError:
        print(f"Model '{model_name}' not found. Please create the model first using deploy_pretrained_model.py")
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
    parser.add_argument('action', choices=['start', 'stop', 'status'], help='Action to perform')
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
