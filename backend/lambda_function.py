import os
import json
import boto3
import logging
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Get the SageMaker endpoint name from environment variable
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'summarizer-endpoint')

def lambda_handler(event, context):
    """
    Lambda function that receives a request with text, passes it to a SageMaker endpoint,
    and returns the summarized text.
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse the incoming request body
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key'
                },
                'body': json.dumps({'error': 'No request body provided'})
            }
        
        # Extract the text to summarize
        if 'text' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key'
                },
                'body': json.dumps({'error': 'No text provided for summarization'})
            }
        
        text = body['text']
        
        # Check if the text is not too short
        if len(text.split()) < 10:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key'
                },
                'body': json.dumps({'error': 'Text is too short to summarize. Please provide a longer text.'})
            }
        
        # Log some stats about the input
        logger.info(f"Text length: {len(text)} characters, {len(text.split())} words")
        
        # Invoke the SageMaker endpoint
        logger.info(f"Invoking SageMaker endpoint: {ENDPOINT_NAME}")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({'text': text})
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        # Log successful summarization
        logger.info("Successfully summarized text")
        
        # Return the summary
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key'
            },
            'body': json.dumps({'summary': result['summary']})
        }
        
    except Exception as e:
        # Log the error
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return an error response
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key'
            },
            'body': json.dumps({'error': f'Error processing request: {str(e)}'})
        }
