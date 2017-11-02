---
 layout: post
 title: Running DynamoDB - (downloadable version)
---

## setup
- Ubuntu 16.04.1 LTS VirtualBox VM : Ubuntu Server
- [Dwonload link](http://docs.aws.amazon.com/amazondynamodb/latest/gettingstartedguide/GettingStarted.Download.html)
- Using Python 2.7.12

## installing DynamoDB
- Extract the tarball
- run : java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -sharedDb -inMemory

## installing Boto3
- sudo apt-get install python-boto3

## Testing
- running example [MoviesCreateTable.py](http://docs.aws.amazon.com/amazondynamodb/latest/gettingstartedguide/GettingStarted.Python.01.html)

> script will through follwing error without a valid AWS credential </br>
> botocore.exceptions.NoCredentialsError: Unable to locate credentials

### creating an AWS credential file
- [Quick link](http://boto3.readthedocs.io/en/latest/guide/quickstart.html)
- save it in ~/.aws/credentials

```
[default]
aws_access_key_id = <KEY>
aws_secret_access_key = <KEY>
region=<REGION>
```
- after this the test script should print confirmation
> Table status: ACTIVE
