# LoRA Web Builder - Configuration Guide

This guide explains how to set up the configuration for the LoRA Web Builder to test your generated LoRAs with Together AI.

## Prerequisites

1. **Together AI API Key**: Get one at [together.ai](https://together.ai)
2. **AWS Account**: For S3 storage of LoRA adapters
3. **S3 Bucket**: For storing LoRA adapter files

## Configuration Setup

### 1. Create Secrets File

Copy the example secrets file and fill in your credentials:

```bash
cp client/config/secrets.example.ts client/config/secrets.ts
```

### 2. Configure API Keys

Edit `client/config/secrets.ts` with your actual credentials:

```typescript
export const secrets = {
  // Together AI API Key - get from https://together.ai
  TOGETHER_API_KEY: 'your_actual_together_ai_api_key',
  
  // AWS Configuration for S3 uploads
  AWS_ACCESS_KEY_ID: 'your_actual_aws_access_key',
  AWS_SECRET_ACCESS_KEY: 'your_actual_aws_secret_key',
  AWS_REGION: 'us-east-1',
  AWS_BUCKET_NAME: 'your-actual-bucket-name',
  
  // Optional: AWS Session Token (if using temporary credentials)
  AWS_SESSION_TOKEN: undefined,
};
```

### 3. Set Up S3 Bucket

Create an S3 bucket for storing LoRA adapters:

```bash
# Using AWS CLI
aws s3 mb s3://your-lora-bucket-name --region us-east-1
```

Or create it through the AWS Console.

### 4. Configure Bucket Permissions

Your S3 bucket needs to allow:
- PutObject (for uploading LoRA files)
- GetObject (for generating presigned URLs)

Example IAM policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::your-lora-bucket-name/*"
    }
  ]
}
```

## Usage

1. **Generate LoRA**: Enter text description and click "Generate LoRA"
2. **Test LoRA**: Once generated, the testing interface will appear
3. **Enter Test Prompt**: Add any prompt to test your LoRA with
4. **Compare Results**: View base model vs LoRA-enhanced responses side-by-side

## Configuration Status

The interface will show:
- ✅ **Configuration complete**: Ready to test LoRAs
- ❌ **Missing credentials**: Check your `secrets.ts` file

## Security Notes

- `secrets.ts` is gitignored and won't be committed to version control
- API keys are only used client-side for direct API calls
- S3 uploads use presigned URLs with 1-hour expiration

## Troubleshooting

### "Missing credentials" error
- Ensure all required fields in `secrets.ts` are filled in
- Check that your AWS credentials have proper S3 permissions

### "Together AI API error"
- Verify your Together AI API key is correct
- Check your Together AI account has sufficient credits

### "S3 upload failed"
- Verify your S3 bucket exists and is accessible
- Check your AWS credentials have PutObject permissions
- Ensure the bucket name matches your configuration

## Development

To run the application:

```bash
npm install
npm run dev
```

The application will be available at `http://localhost:5173`. 