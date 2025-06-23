# Stream Example

## Install AWS CLI and configure
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

```bash
aws configure
# AWS_ACCESS_KEY_ID = (Create new user in IAM AWS)
# AWS_SECRET_ACCESS_KEY = (Create new user in IAM AWS)
# AWS_DEFAULT_REGION = (Default Region AWS)
```

## Test kinesis aws

```bash
KINESIS_STREAM_INPUT=mlops_ride_events
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --data "Hello, this is a test."
```

## Test AWS Kinesis - Lambda
```bash
KINESIS_STREAM_INPUT=mlops_ride_events 
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --data "Hello, this is a test."
```
Test Lambda $ Kinesis

```bash
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 156
    }'
```

Event example output

```bash
{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::090719694989:role/lambda-kinesis-role",
            "awsRegion": "eu-west-2",
            "eventSourceARN": "arn:aws:kinesis:eu-west-2:090719694989:stream/mlops_ride_events"
        }
    ]
```

## Reading from stream

```bash
KINESIS_STREAM_OUTPUT='mlops_ride_events'
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type TRIM_HORIZON \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
)
```

## Runing test in local

```bash
# activate enviroment
pipenv shell
python test.py
```

## Build and run dokcer image

```bash
docker build -t stream-model-duration:v1 .

docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="mlops_ride_predictions" \
    -e RUN_ID="m-ea1dfabd247448bbaaba443d31ce73b9" \
    -e TEST_RUN="True" \
    -e AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)" \
    -e AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)" \
    -e AWS_DEFAULT_REGION="$(aws configure get region)" \
    stream-model-duration:v1
```

## Login aws cli in ECR

```bash
aws ecr get-login-password --region us-east-2 | \
docker login --username AWS --password-stdin 090719694989.dkr.ecr.us-east-2.amazonaws.com
```
```bash
090719694989.dkr.ecr.us-east-2.amazonaws.com
│          │        └────── Region: us-east-2 (Ohio)
│          └─────────────── Service: ECR (Elastic Container Registry)
└────────────────────────── ID daws account
```

## Push image to ECR

```bash
REMOTE_URI="090719694989.dkr.ecr.us-east-2.amazonaws.com/duration-model"
REMOTE_TAG="v1"
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE="stream-model-duration:v1"
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
```