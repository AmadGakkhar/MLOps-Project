# MLOps-Project

We will divide this project in the following parts

1.  Project and Environment Setup
2.  Database Setup
3.  EDA(Exploratory Data Analysis), Feature Engineering
4.  Data Ingestion Module
5.  Data Validation
6.  Model Training and Evaluation
7.  Hyperparameter Tuning
8.  Docker Image Creation
9.  Setting Up Workflows using Github Actions (CI/CD)
10.  Deployment on AWS


## Deployment Steps:

1. Build Docker Image of your Source Code
2. Push Docker Image to ECR (Elastic Container Registry)
3. Launch EC2 instance
4. Pull Image from ECR to EC2 instance
5. Launch Docker Image in EC2 instance


## AWS CI/CD Deployment with Github Actions

1. Login to AWS Console
2. Create IAM (Identity Access Manager) User
    Poilcy:
    AmazonEC2ContainerRegistryFullAccess
    AmazonEC2FullAccess

3. Create ECR repo to store Docker Image.
    - save the uri

4. Create EC2 Machine (Ubuntu)
5. Open EC2 and install Docker

    sudo apt-get update -y

    sudo apt-get upgrade


    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

6. Configure EC2 as self hosted runner
    Github
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

7.  Setup Github Secrets

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


