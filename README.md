# Creating and Deploying a Custom Model in Sagemaker
David Hren

In this blog, we will give an example of defining machine learning model in Python and then deploying it using Amazon Sagemaker.

### Overview of Sagemaker Models

SageMaker uses Docker containers to compartmentalize ML algorithms; this container approach allows SageMaker to offer a wide range of readily available algorithms for common use-cases while remaining flexible enough to support models developed using common libraries or even a completely custom written model. The model containers can be used on three basic levels:

1.	Pre-built Algorithms – fixed class of algorithms fully maintained by AWS
2.	"Script Mode" – allows popular ML frameworks to be utilized via a script 
3.	"Container Mode" – allows for a fully customized ML algorithm to be used

These modes offer various degrees of both complexity and ease of use. 

**Image here**

First, we will give a brief rundown of each mode, but we will focus showing a step-by-step process to use container mode to deploy a machine learning model. If you are new to using Sagemaker, you can find a series of [deep dive videos](https://www.youtube.com/playlist?list=PLhr1KZpdzukcOr_6j_zmSrvYnLUtgqsZz) produced by AWS helpful. 

In addition to the standard [AWS SDKs](https://aws.amazon.com/tools/), Amazon also has a higher level Python package (the [Sagemaker Python SDK](https://sagemaker.readthedocs.io/en/stable/#)) for training and deploying models using Sagemaker, which we will use here.

##### Pre-built Algorithms
SageMaker offers following pre-built algorithms that can tackle a wide range of problem types and use cases. AWS maintains all of the containers associated with these algorithms. You can find the full list of available algorithms and read more about each one on the [SageMaker documentation page](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html).

##### Script Mode
Script mode allows you to write Python scripts against commonly used machine learning [frameworks](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html). AWS still maintains the underlying container hosting whichever framework you choose, and your script is embedded into the container and used to direct the logic during runtime. In order for you script to be compatible with the AWS maintained container, the script must meet certain design requirements.

##### Container Mode
Container mode allows you to to use custom logic to define a model and deploy it into the SageMaker ecosystem; in this mode you for maintaining both the container and the underlying logic it implements. This mode is the most flexible and can let you access the many Python libraries and machine learning tools available. In order for the container to be compatible with SageMaker, your container must meet certain design requirements. This can be accomplished in one of two ways:

1. Define your custom container by extending one of the existing ones maintained by AWS
2. Use the [SageMaker Containers Library](https://github.com/aws/sagemaker-training-toolkit) to define your container.

We will focus on using method 1. here, but AWS reall has made every effort to make it as easy as possible to use your own custom logic within Sagemaker.

After designing you container, you must uploaded it to the AWS Elastic Container Registry (ECR); this is the model image you will point SageMaker to when training or deploying a model.

### Outline of the Steps

Here we will outline the basic steps involved in creating and deploying a custom model in Sagemaker. 

1. Define the logic of the machine learning model
2. Define the model image
3. Train and deploy the model image

### Defining the Logic of the Model

 

### Defining the Model Image

 

### Training and Deploying the Custom Model
