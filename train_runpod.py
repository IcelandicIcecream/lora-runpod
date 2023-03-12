import predict
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
    'instance_data': {
        'type': str,
        'required': True
    },
    'task': {
        'type': str,
        'required': False,
        'default': "face"
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None,
    },
    'resolution': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'placeholder_tokens': {
        'type': str,
        'required': False,
        'default': "<s1>|<s2>",
    },
}

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''

    job_input = job['input']

     # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    lora_paths = MODEL.predict(
        instance_data=job_input["instance_data"],
        task=job_input["task"],
        seed=job_input["seed"],
        resolution=job_input["resolution"],
        placeholder_tokens=job_input["placeholder_tokens"],
    )

    job_output = {
        "lora_model": lora_paths
    }

    return job_output

runpod.serverless.start({"handler": run})