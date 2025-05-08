import boto3
import sys
import time

def delete_all_endpoints():
    """Delete all SageMaker endpoints in the AWS account"""
    sm_client = boto3.client('sagemaker')
    
    print("Retrieving all SageMaker endpoints...")
    endpoints = sm_client.list_endpoints()
    
    if not endpoints['Endpoints']:
        print("No endpoints found.")
        return
    
    print(f"Found {len(endpoints['Endpoints'])} endpoints.")
    
    # Option to confirm deletion
    if len(sys.argv) <= 1 or sys.argv[1] != "--force":
        confirm = input(f"Delete all {len(endpoints['Endpoints'])} endpoints? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Delete each endpoint
    for endpoint in endpoints['Endpoints']:
        endpoint_name = endpoint['EndpointName']
        print(f"Deleting endpoint: {endpoint_name}")
        
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"  ✓ Deletion initiated for {endpoint_name}")
        except Exception as e:
            print(f"  ✗ Error deleting {endpoint_name}: {str(e)}")
    
    print("Deletion process initiated for all endpoints.")
    print("Checking for remaining endpoints...")
    
    # Brief wait to allow AWS to process deletions
    time.sleep(5)
    
    # Check final status
    remaining = sm_client.list_endpoints()
    if not remaining['Endpoints']:
        print("All endpoints have been deleted.")
    else:
        print(f"{len(remaining['Endpoints'])} endpoints still being deleted. Run script again later to check.")

if __name__ == "__main__":
    delete_all_endpoints()
