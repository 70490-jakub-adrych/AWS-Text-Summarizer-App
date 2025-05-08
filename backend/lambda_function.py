import json
import boto3
import os
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Log the entire event to diagnose the structure
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Extract HTTP method - API Gateway can send it in different ways
    http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
    
    # If we still can't determine the method, log this fact and try to infer it
    if not http_method:
        logger.warning("Could not find httpMethod in event, trying to infer")
        # If there's a 'body' but no specific method, assume POST
        if event.get('body'):
            http_method = 'POST'
            logger.info("Inferred POST method due to presence of body")
        else:
            # If no body, default to handling as POST to be safe
            http_method = 'POST'
            logger.info("Defaulting to POST method")
    
    # Handle OPTIONS request for CORS
    if http_method == 'OPTIONS':
        logger.info("Handling OPTIONS request for CORS")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({})
        }
    
    # Handle POST request for summarization
    if http_method == 'POST':
        try:
            # Get the request body - with special handling for different API Gateway formats
            body_str = event.get('body', '{}')
            logger.info(f"Raw request body type: {type(body_str)}")
            logger.info(f"Raw request body: {body_str}")
            
            # API Gateway might send the body in different formats:
            # 1. As a JSON string in the 'body' field
            # 2. As a base64-encoded string that needs to be decoded
            # 3. As an already parsed Python dictionary
            # 4. As a JSON string with escaped quotes that needs to be unescaped
            
            # Handle case when body is already a dict
            if isinstance(body_str, dict):
                body = body_str
                logger.info("Body is already a dictionary")
            else:
                # Try to handle various string formats
                try:
                    # Check if the body is a JSON string with escaped quotes
                    if isinstance(body_str, str) and body_str.startswith('"') and body_str.endswith('"'):
                        body_str = json.loads(body_str)  # This will unescape it
                        logger.info("Unescaped JSON string body")
                    
                    # Now parse it as JSON
                    body = json.loads(body_str)
                    logger.info("Successfully parsed body as JSON")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse request body as JSON: {e}")
                    # If it's not valid JSON, try to use it directly as text
                    if isinstance(body_str, str) and body_str:
                        body = {"text": body_str}
                        logger.info("Using raw body string as text")
                    else:
                        body = {}
                        logger.error("Could not extract text from body")
            
            # Log the parsed body to verify structure
            logger.info(f"Parsed body type: {type(body)}")
            logger.info(f"Parsed body: {json.dumps(body) if isinstance(body, dict) else str(body)}")
            
            # Try different ways to extract the text
            text = None
            
            # Method 1: Direct extraction from body dictionary
            if isinstance(body, dict) and 'text' in body:
                text = body['text']
                logger.info("Found text in body dictionary")
            
            # Method 2: If body is a simple string, use it directly
            elif isinstance(body, str) and body:
                text = body
                logger.info("Using body string directly as text")
            
            # Method 3: Check for nested structures
            elif isinstance(body, dict):
                # Look for nested text field
                for key, value in body.items():
                    if isinstance(value, dict) and 'text' in value:
                        text = value['text']
                        logger.info(f"Found text in nested structure under key: {key}")
                        break
            
            # If still no text found, check if it might be in the event directly
            if not text and event.get('text'):
                text = event['text']
                logger.info("Found text directly in event")
            
            # Log what we found or didn't find
            if text:
                logger.info(f"Extracted text (first 50 chars): {text[:50]}...")
                logger.info(f"Text length: {len(text)}")
            else:
                logger.error("No text found in any expected location in the request")
                logger.error(f"Event keys: {list(event.keys())}")
                if isinstance(body, dict):
                    logger.error(f"Body keys: {list(body.keys())}")
                
                return {
                    'statusCode': 400,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                        'Access-Control-Allow-Methods': 'OPTIONS,POST'
                    },
                    'body': json.dumps({'error': 'No text provided for summarization'})
                }
            
            # Check if text is empty after trimming whitespace
            if not text.strip():
                logger.error("Text is empty or whitespace only")
                return {
                    'statusCode': 400,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                        'Access-Control-Allow-Methods': 'OPTIONS,POST'
                    },
                    'body': json.dumps({'error': 'No text provided for summarization'})
                }
            
            # Initialize SageMaker runtime client with increased timeout
            client = boto3.client(
                'sagemaker-runtime',
                config=boto3.session.Config(
                    read_timeout=25,  # Increase read timeout to 25 seconds
                    connect_timeout=10,  # Increase connection timeout to 10 seconds
                    retries={'max_attempts': 2}  # Add retry behavior
                )
            )
            
            # Log before calling endpoint
            endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT', 'huggingface-pytorch-inference-2025-05-08-18-46-16-895')
            logger.info(f"Calling SageMaker endpoint: {endpoint_name}")
            
            # Call the endpoint with text length optimization
            # For very long texts, consider truncating to improve response time
            max_input_length = 5000  # Maximum characters to send to the model
            if len(text) > max_input_length:
                logger.info(f"Text too long ({len(text)} chars), truncating to {max_input_length} chars")
                text = text[:max_input_length] + "..."
            
            # Call the endpoint
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps({"inputs": text}),
                ContentType='application/json'
            )
            
            # Process the response
            result_body = response['Body'].read()
            logger.info(f"Response from SageMaker: {result_body}")
            result = json.loads(result_body)
            
            # Format the response for the frontend
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
            else:
                summary = result.get("summary_text", "")
            
            logger.info(f"Generated summary with length: {len(summary)}")
            logger.info(f"Summary (first 50 chars): {summary[:50] if summary else 'No summary generated'}")
            
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST',
                    'Cache-Control': 'no-cache'  # Prevent caching of results
                },
                'body': json.dumps({'summary': summary})
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            
            # Check if it's a timeout error
            error_message = str(e).lower()
            if 'timeout' in error_message or 'timed out' in error_message:
                return {
                    'statusCode': 504,  # Gateway Timeout
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                        'Access-Control-Allow-Methods': 'OPTIONS,POST'
                    },
                    'body': json.dumps({
                        'error': 'The summarization request timed out. Please try with a shorter text or try again later.',
                        'details': str(e)
                    })
                }
            
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({'error': f'Failed to summarize text: {str(e)}'})
            }
    
    # Log the unexpected method
    logger.error(f"Unsupported HTTP method: {http_method}")
    return {
        'statusCode': 405,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps({'error': 'Method not allowed'})
    }