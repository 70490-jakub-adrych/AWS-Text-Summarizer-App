@echo off
echo Installing AWS CLI...
echo Please wait while the installer downloads...
powershell -Command "& {Invoke-WebRequest -Uri 'https://awscli.amazonaws.com/AWSCLIV2.msi' -OutFile 'AWSCLIV2.msi'; Start-Process msiexec.exe -Wait -ArgumentList '/i AWSCLIV2.msi /quiet'}"
echo AWS CLI installation complete.
echo.
echo Please configure AWS credentials by running: aws configure
echo Then build your React app with: npm run build
echo Finally deploy with: aws s3 sync build/ s3://textsummarizerbucket-test --acl public-read

