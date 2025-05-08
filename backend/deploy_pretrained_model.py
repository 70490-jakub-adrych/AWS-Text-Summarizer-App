from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

role = get_execution_role()
hub = {
  'HF_MODEL_ID': 'sshleifer/distilbart-cnn-6-6',
  'HF_TASK':     'summarization'
}

huggingface_model = HuggingFaceModel(
    transformers_version='4.26',  # or newer
    pytorch_version='1.13',       # or newer
    py_version='py39',
    env=hub,
    role=role
)
# Deploy on a suitable instance (see below)
predictor = huggingface_model.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")