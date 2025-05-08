# AWS Text Summarization Application

This application uses AWS services and React to provide text summarization capabilities. Users can sign in using Amazon Cognito, input text, and receive AI-generated summaries powered by a fine-tuned transformer model on SageMaker.

## Architecture Overview

- **Frontend**: React application with Cognito authentication
- **Backend**: 
  - SageMaker endpoint running a fine-tuned BART model
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

The application uses an existing Cognito User Pool. Here are the current settings:

- **Region**: eu-north-1
- **User Pool ID**: eu-north-1_O4CIZM2aJ
- **App Client ID**: 46g5ig1s03klm5guub0h4qkv2e
- **Domain**: https://eu-north-1o4cizm2aj.auth.eu-north-1.amazoncognito.com
- **Callback URL**: https://d3cuf8vpsz3a8w.cloudfront.net/

To create a new Cognito User Pool (if necessary):

1. Navigate to the Amazon Cognito console
2. Click "Create a User Pool"
3. Follow the wizard to set up authentication flows
4. Under App integration, add an App client and configure the domain name
5. Set the callback URL to your frontend URL
6. Note the User Pool ID and App Client ID for your config.js file

### 2. SageMaker Model Training and Deployment

The model training is handled through SageMaker using a Jupyter notebook:

1. Navigate to Amazon SageMaker in the AWS console
2. Create a new notebook instance with at least ml.t3.medium (free tier eligible)
   - Choose **Python 3 (ipykernel)** as your kernel
   - Ensure the instance role has access to S3 and necessary SageMaker permissions
3. Upload the files from the `backend/` directory to the notebook instance:
   - Click the "Upload" button in the Jupyter interface
   - Select `train.py`, `preprocess.py`, and any other Python files from your local `backend/` directory
   - Create a `backend/` directory in your notebook instance and place the uploaded files there
4. Upload `notebooks/model_training.ipynb` to your instance
5. Open the notebook and ensure you select the Python 3 (ipykernel) kernel
6. Run each cell in the notebook sequentially to:
   - Preprocess the CNN/DailyMail dataset (using a very small subset for faster training)
   - Fine-tune the BART model on free tier compatible instances
   - Deploy the model to a SageMaker endpoint

> **Note:** The notebook has been configured to use AWS Free Tier compatible instance types:
> - `ml.t3.medium` for processing jobs
> - `ml.m5.xlarge` for training (free tier includes 50 hours)
> - `ml.m5.xlarge` for inference (free tier includes 125 hours)

> **Note:** If you encounter errors related to instance types or image URIs, make sure all instance type variables are properly defined and that you're providing all required parameters in the `sagemaker.image_uris.retrieve()` function, including `image_scope`, `py_version`, and `instance_type`.

> **Note:** The preprocessing script automatically installs required Python packages (`datasets`, `pandas`, `transformers`) at runtime in the SageMaker container. This may take a few minutes during the first execution.

This process may take several hours and will incur AWS charges for SageMaker usage. GPU-enabled instances (p3 family) are recommended for faster training.

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
   - CognitoUserPoolId parameter (`eu-north-1_O4CIZM2aJ`)
   - SageMakerEndpointName parameter (`summarizer-endpoint`)

5. After deployment completes, note the API Gateway URL (this will be your API_ENDPOINT)

### 4. Frontend Configuration and Deployment

1. Ensure the `src/config.js` file has all the correct values:
   ```js
   export const AWS_REGION = 'eu-north-1';
   export const USER_POOL_ID = 'eu-north-1_O4CIZM2aJ';
   export const USER_POOL_CLIENT_ID = '46g5ig1s03klm5guub0h4qkv2e';
   export const API_ENDPOINT = 'https://your-api-gateway-url.execute-api.eu-north-1.amazonaws.com/prod';
   export const COGNITO_DOMAIN = 'https://eu-north-1o4cizm2aj.auth.eu-north-1.amazoncognito.com';
   export const REDIRECT_URI = 'https://d3cuf8vpsz3a8w.cloudfront.net/';
   export const S3_BUCKET = 'textsummarizerbucket-test';
   export const SAGEMAKER_ENDPOINT = 'summarizer-endpoint';
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run locally for testing:
   ```
   npm start
   ```

4. Build and deploy to S3:
   ```
   npm run deploy
   ```

5. Create CloudFront invalidation to ensure the latest version is served:
   ```
   aws cloudfront create-invalidation --distribution-id E1IVRZ97YG05C2 --paths "/*"
   ```

## Security Considerations

- The `src/config.js` file is added to .gitignore to prevent secrets from being committed
- API Gateway endpoints are secured with Cognito authorizers
- CORS is configured to allow only necessary origins
- All requests use HTTPS

## CI/CD Integration (Optional)

To set up continuous deployment:

1. Create a CodeCommit repository for your code
2. Set up an AWS CodePipeline with:
   - CodeCommit as the source
   - CodeBuild to build the frontend and backend
   - CloudFormation for deploying the backend resources
   - S3 deployment for the frontend

## Troubleshooting

- **Authentication Issues**: Check Cognito configuration and ensure redirect URIs match
- **API Permissions**: Verify the Cognito authorizer is correctly set up in API Gateway
- **Model Performance**: Check CloudWatch logs for the SageMaker endpoint and Lambda function

## License

This project is licensed under the MIT License - see the LICENSE file for details.
