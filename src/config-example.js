// !!! IMPORTANT: This is a template configuration file. RENAME TO config.js AND FILL IN YOUR VALUES!!!

// This file contains sensitive information. Do not share it publicly.

// AWS Configuration
export const AWS_REGION = 'region'; // e.g., 'eu-north-1'
export const USER_POOL_ID = 'user-pool-id'; // e.g., 'eu-north-1_CFDS5tsd'
export const USER_POOL_CLIENT_ID = 'User-Pool-Client-ID'; // e.g., '46g5ig1ssdfsdfuub0h4qkv2e'
export const REDIRECT_URI = 'https://(unique_id).cloudfront.net/'; // e.g., 'https://d3cuaaaasz3a8w.cloudfront.net/'

// API Configuration
// This should be the URL from your API Gateway deployment
export const API_ENDPOINT = 'https://(unique_id).execute-api.eu-north-1.amazonaws.com/prod'; // eg., 'https://sbaaaaaaa.execute-api.eu-north-1.amazonaws.com/prod'

// SageMaker Configuration
export const SAGEMAKER_ENDPOINT = 'your_endpoint_name'; // e.g., 'huggingface-pytorch-inference-2025-05-06-18-46-20-895'
