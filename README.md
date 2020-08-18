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

In addition to the standard [AWS SDKs](https://aws.amazon.com/tools/), Amazon also has a higher level Python package (the [Sagemaker Python SDK](https://sagemaker.readthedocs.io/en/stable/#)) for training and deploying models using Sagemaker, which we will use here.

##### Pre-built Algorithms
SageMaker offers following pre-built algorithms that can tackle a wide range of problem types and use cases. AWS maintains all of the containers associated with these algorithms. You can find the full list of available algorithms and read more about each one on the [SageMaker documentation page](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html).

##### Script Mode
Script mode allows you to write Python scripts against commonly used machine learning frameworks (see the )

### Defining the Logic of the Model

 

### Defining the Model Image

 

### Training and Deploying the Custom Model
