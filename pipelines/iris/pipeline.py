"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session
# import mlflow
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import IntegerParameter, HyperparameterTuner


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="IrisPackageGroup",
    pipeline_name="IrisPipeline",
    base_job_prefix="Iris",
    processing_instance_type="ml.m5.large",
    training_instance_type="ml.m5.large",
    source_code_commit="",
    source_pipeline_trigger="",
):
    """Gets a SageMaker ML Pipeline instance working on iris data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", 
        default_value=1
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    mlflow_tracking_uri = ParameterString(
        name='MLflowTrackingURI',
        default_value='',
    )
    mlflow_experiment_name = ParameterString(
        name='MLflowExperimentName',
        default_value='sagemaker-mlflow-iris',
    )
    mlflow_model_name = ParameterString(
        name='MLflowModelName',
        default_value='sklearn-iris',
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-iris-prepare-data",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_process = ProcessingStep(
        name="PrepareIrisData",
        processor=sklearn_processor,
        code=os.path.join(BASE_DIR, 'prepare_data.py'),
        job_arguments=['--output-dir', '/opt/ml/processing/data'],
        outputs=[
            ProcessingOutput(
                output_name='data',
                source='/opt/ml/processing/data',
            )
        ]
    )

    # training step for generating model artifacts
    hyperparameters = {
        'mlflow-tracking-uri': mlflow_tracking_uri,
        'mlflow-experiment-name': mlflow_experiment_name,
        'mlflow-model-name': mlflow_model_name,
        'source-commit': source_code_commit,
        'source-trigger': source_pipeline_trigger,
        'train-file': 'iris_train.csv',
        'test-file': 'iris_test.csv',
    }

    metric_definitions = [
        {'Name': 'accuracy', 'Regex': "metric_accuracy=([0-9.]+).*$"},
        {'Name': 'f1', 'Regex': "metric_f1=([0-9.]+).*$"},
    ]

    estimator = SKLearn(
        entry_point='train.py',
        source_dir=os.path.join(BASE_DIR, 'source_dir'),
        role=role,
        metric_definitions=metric_definitions,
        hyperparameters=hyperparameters,
        instance_count=1,
        instance_type=training_instance_type,
        framework_version='1.0-1',
        base_job_name=f"{base_job_prefix}/sklearn-iris-train",
        sagemaker_session=pipeline_session,
        disable_profiler=True
    )
    
    hyperparameter_ranges = {
        'max-leaf-nodes': IntegerParameter(2, 3),
        'max-depth': IntegerParameter(2, 3),
    }
    
    objective_metric_name = 'accuracy'
    objective_type = 'Maximize'
    
    hp_tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=4,
        max_parallel_jobs=4,
        objective_type=objective_type,
        base_tuning_job_name=f"{base_job_prefix}/sklearn-iris-tune",
    )
    
    step_tuning = TuningStep(
        name = "IrisTuning",
        tuner = hp_tuner,
        inputs = {
            "input": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["data"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            mlflow_tracking_uri,
            mlflow_experiment_name,
            mlflow_model_name
        ],
        steps=[step_process, step_tuning],
        sagemaker_session=pipeline_session,
    )
    return pipeline
