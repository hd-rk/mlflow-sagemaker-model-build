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
        framework_version="0.23-1",
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
        # 'tracking_uri': tracking_uri,
        # 'experiment_name': experiment_name,
        # 'registered_model_name': registered_model_name,
        'train-file': 'iris_train.csv',
        'test-file': 'iris_test.csv',
        # 'max-leaf-nodes': 4,
        # 'max-depth': 2,
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
        framework_version='0.23-1',
        base_job_name=f"{base_job_prefix}/sklearn-iris-train",
        # sagemaker_session=pipeline_session,
        disable_profiler=True
    )
    
    # step_train = TrainingStep(
    #     name="TrainModel",
    #     estimator=estimator,
    #     inputs={
    #         "input": TrainingInput(
    #             s3_data=step_process.properties.ProcessingOutputConfig.Outputs["data"].S3Output.S3Uri,
    #             content_type="text/csv",
    #         ),
    #     },
    # )
    
    hyperparameter_ranges = {
        'max-leaf-nodes': IntegerParameter(2, 5),
        'max-depth': IntegerParameter(2, 5),
    }
    
    objective_metric_name = 'accuracy'
    objective_type = 'Maximize'
    
    hp_tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=16,
        max_parallel_jobs=2,
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
#     model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
#     image_uri = sagemaker.image_uris.retrieve(
#         framework="xgboost",
#         region=region,
#         version="1.0-1",
#         py_version="py3",
#         instance_type=training_instance_type,
#     )
#     xgb_train = Estimator(
#         image_uri=image_uri,
#         instance_type=training_instance_type,
#         instance_count=1,
#         output_path=model_path,
#         base_job_name=f"{base_job_prefix}/abalone-train",
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     xgb_train.set_hyperparameters(
#         objective="reg:linear",
#         num_round=50,
#         max_depth=5,
#         eta=0.2,
#         gamma=4,
#         min_child_weight=6,
#         subsample=0.7,
#         silent=0,
#     )
#     step_args = xgb_train.fit(
#         inputs={
#             "train": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "train"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#             "validation": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "validation"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#         },
#     )
#     step_train = TrainingStep(
#         name="TrainAbaloneModel",
#         step_args=step_args,
#     )

#     # processing step for evaluation
#     script_eval = ScriptProcessor(
#         image_uri=image_uri,
#         command=["python3"],
#         instance_type=processing_instance_type,
#         instance_count=1,
#         base_job_name=f"{base_job_prefix}/script-abalone-eval",
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     step_args = script_eval.run(
#         inputs=[
#             ProcessingInput(
#                 source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#                 destination="/opt/ml/processing/model",
#             ),
#             ProcessingInput(
#                 source=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "test"
#                 ].S3Output.S3Uri,
#                 destination="/opt/ml/processing/test",
#             ),
#         ],
#         outputs=[
#             ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
#         ],
#         code=os.path.join(BASE_DIR, "evaluate.py"),
#     )
#     evaluation_report = PropertyFile(
#         name="AbaloneEvaluationReport",
#         output_name="evaluation",
#         path="evaluation.json",
#     )
#     step_eval = ProcessingStep(
#         name="EvaluateAbaloneModel",
#         step_args=step_args,
#         property_files=[evaluation_report],
#     )

#     # register model step that will be conditionally executed
#     model_metrics = ModelMetrics(
#         model_statistics=MetricsSource(
#             s3_uri="{}/evaluation.json".format(
#                 step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
#             ),
#             content_type="application/json"
#         )
#     )
#     model = Model(
#         image_uri=image_uri,
#         model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     step_args = model.register(
#         content_types=["text/csv"],
#         response_types=["text/csv"],
#         inference_instances=["ml.t2.medium", "ml.m5.large"],
#         transform_instances=["ml.m5.large"],
#         model_package_group_name=model_package_group_name,
#         approval_status=model_approval_status,
#         model_metrics=model_metrics,
#     )
#     step_register = ModelStep(
#         name="RegisterAbaloneModel",
#         step_args=step_args,
#     )

#     # condition step for evaluating model quality and branching execution
#     cond_lte = ConditionLessThanOrEqualTo(
#         left=JsonGet(
#             step_name=step_eval.name,
#             property_file=evaluation_report,
#             json_path="regression_metrics.mse.value"
#         ),
#         right=6.0,
#     )
#     step_cond = ConditionStep(
#         name="CheckMSEAbaloneEvaluation",
#         conditions=[cond_lte],
#         if_steps=[step_register],
#         else_steps=[],
#     )

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
        # steps=[step_process, step_train, step_eval, step_cond],
        steps=[step_process, step_tuning],
        sagemaker_session=pipeline_session,
    )
    return pipeline
