ARG REGION=us-east-1

FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3

ENV PATH="/opt/ml/code:${PATH}"

COPY /code /opt/ml/code
COPY gam_model/dist/gam_model-0.0.1-py3-none-any.whl /opt/gam_model-0.0.1-py3-none-any.whl

RUN pip install -r /opt/ml/code/requirements.txt /opt/gam_model-0.0.1-py3-none-any.whl

ENV SAGEMAKER_PROGRAM model_logic.py