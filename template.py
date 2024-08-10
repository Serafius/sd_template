import base64
from koldstart import isolated, CloudKeyCredentials, KoldstartHost, cached


@cached
def setup():
    from pathlib import Path
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    import time

    cache_dir = Path("/data/vaha/illustration").expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_id = "Serafius/illustrationmodelbyvaha"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token="")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        use_auth_token="",
        torch_dtype=torch.float16,
        revision="main",
        cache_dir=cache_dir,
    )
    start = time.time()
    pipe = pipe.to("cuda")
    print("Time: " , time.time() - start)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None

    return pipe


requirements = [
    "accelerate",
    "diffusers[torch]>=0.10",
    "ftfy",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "safetensors",
    "xformers==0.0.16",
    "huggingface_hub ",
    "pillow",
    "opencv-python",
    "numpy",
    "controlnet_aux",
    "matplotlib",
    "mediapipe"
]


@isolated(
    requirements=requirements, machine_type="GPU", keep_alive=30
)
def generate_image(prompt=None, seed=None, negative_prompt=None, width=None, height=None):
    import base64
    import io
    import torch
    pipe = setup()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt, guidance_scale=9, num_inference_steps=40, width=width , height=height, negative_prompt=negative_prompt, generator=generator
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="png")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes)

def write_image():
    import base64
    import sys
    import boto3
    import random
    print("Hey")
    seed = int(sys.argv[10])
    if seed < 1000:
        seed = random.randint(10**3,10**10)
    width = 1024
    height = 1024
    if sys.argv[9] == "vertical":
        width = 768
        height = 1344
    if sys.argv[9] == "horizontal":
        width = 1344
        height = 768    
    file = generate_image(
        prompt=sys.argv[1] + ",illustration,illustrationmodelbyvahastudio",
        seed=seed,
        negative_prompt=sys.argv[8],
        width = width,
        height = height
    )

    s3 = boto3.client('s3', "us-east-1", aws_access_key_id="", aws_secret_access_key="")
    metadata = {
        "Content-Disposition": f"attachment; filename=Image-{sys.argv[2]}-{sys.argv[3]}.png"
    }
    s3.put_object(Body=base64.b64decode(file),Bucket="stock-image",Key=f"testtemp/Image-{sys.argv[2]}-{int(sys.argv[3]) + seed}.png", ACL="public-read", ContentType='image/png')
    print(f'https://media.stockimg.ai/testtemp/Image-{sys.argv[2]}-{int(sys.argv[3]) + seed}.png')
    print(seed)


write_image()
