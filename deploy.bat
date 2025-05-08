@echo off
echo Building React application...
call npm run build
echo Build complete. Deploying to S3...
aws s3 sync build/ s3://textsummarizerbucket-test --acl public-read
echo Deployment complete!

REM If you have CloudFront, uncomment the next line and add your distribution ID
REM aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
