# AWS Text Summarization Application

This application uses AWS services and React to provide text summarization capabilities. Users can sign in using Amazon Cognito, input text, and receive AI-generated summaries powered by a transformer model on SageMaker.

## Architecture Overview

- **Frontend**: React application with Cognito authentication
- **Backend**: 
  - SageMaker endpoint running a distilBART model
  - Lambda function for processing requests
  - API Gateway for secure access
  - Cognito for authentication

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Node.js (v14+) and npm
- Python 3.8+ (for backend development)
- Docker (optional, for local testing)

## Setup Instructions

### 1. Cognito Configuration

To create a new Cognito User Pool:

1. Navigate to the Amazon Cognito console
2. Click "Create a User Pool"
3. Follow the wizard to set up authentication flows
4. Under App integration, add an App client and configure the domain name
5. Set the callback URL to your frontend URL
6. Note the User Pool ID and App Client ID for your config.js file

### 2. SageMaker Model Deployment

The model can be deployed through SageMaker using a Jupyter notebook or our provided script:

1. Navigate to Amazon SageMaker in the AWS console
2. Create a new notebook instance with at least ml.t3.medium (free tier eligible)
   - Choose **Python 3 (ipykernel)** as your kernel
   - Ensure the instance role has access to S3 and necessary SageMaker permissions
3. Upload the files from the `backend/` directory to the notebook instance
4. Upload `notebooks/deploy_pretrained_model.ipynb` to your instance
5. Open the notebook and run each cell sequentially to deploy the model

Alternatively, you can use the script in the backend folder:

```bash
cd backend
python deploy_pretrained_model.py
```

> **Note:** The script and notebook are configured for AWS Free Tier compatible instance types:
> - `ml.m5.xlarge` for inference (free tier includes 125 hours)

> **Note:** Remember that running a SageMaker endpoint will incur AWS charges. Use the manage_endpoint.py script to control costs:
> ```bash
> # Check endpoint status
> python manage_endpoint.py status
> 
> # Start endpoint
> python manage_endpoint.py start
> 
> # Stop endpoint (deletes the endpoint to prevent charges)
> python manage_endpoint.py stop
> ```

### 3. Lambda and API Gateway Deployment

The serverless backend can be deployed using AWS SAM:

1. Navigate to the `backend/` directory
2. Make sure AWS SAM CLI is installed
3. Run the following commands:
   ```
   sam build
   sam deploy --guided
   ```
4. Follow the prompts, providing:
   - Stack name (e.g., `text-summarizer-stack`)
   - AWS region (e.g., `eu-north-1`)
   - CognitoUserPoolId parameter (from step 1)
   - SageMakerEndpointName parameter (from step 2)

5. After deployment completes, note the API Gateway URL (this will be your API_ENDPOINT)

### 4. Frontend Configuration and Deployment

1. Create a `src/config.js` file based on the example template:
   ```bash
   cp src/config-example.js src/config.js
   ```

2. Edit the `src/config.js` file with your own values for:
   - AWS_REGION
   - USER_POOL_ID
   - USER_POOL_CLIENT_ID
   - API_ENDPOINT
   - REDIRECT_URI
   - SAGEMAKER_ENDPOINT

3. Install dependencies:
   ```
   npm install
   ```

4. Run locally for testing:
   ```
   npm start
   ```

5. Build for deployment:
   ```
   npm run build
   ```

6. Deploy to S3 (replace BUCKET_NAME with your bucket):
   ```
   aws s3 sync build/ s3://BUCKET_NAME
   ```

7. Create CloudFront invalidation (replace DISTRIBUTION_ID with your distribution ID):
   ```
   aws cloudfront create-invalidation --distribution-id DISTRIBUTION_ID --paths "/*"
   ```

## Security Considerations

- The `src/config.js` file is added to .gitignore to prevent secrets from being committed
- API Gateway endpoints are secured with Cognito authorizers
- CORS is configured to allow only necessary origins
- All requests use HTTPS

## Troubleshooting

- **Authentication Issues**: Check Cognito configuration and ensure redirect URIs match
- **API Permissions**: Verify the Cognito authorizer is correctly set up in API Gateway
- **Model Performance**: Check CloudWatch logs for the SageMaker endpoint and Lambda function

## License

This project is licensed under the MIT License - see the LICENSE file for details.
