from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from diffusers import StableDiffusionPipeline
import wandb
import argparse 


# Adapted from PickScore example
def calc_scores(prompt, images, processor, model, device):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    
    return scores.cpu().tolist()


def score(args):

    # Load the dataset
    pap = load_dataset("yuvalkirstain/pickapic_v1_no_images")
    prompts = pap['validation_unique']['caption']

    # Load score model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_pipeline_path, 
        torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.enable_xformers_memory_efficient_attention()

    # Start logging
    wandb.init(project='pickapic_eval', config=args)

    # Score on N prompts, default settings
    scores = []
    for i, prompt in enumerate(prompts[:args.n_prompts]):
        g = torch.Generator().manual_seed(i)
        pil_images = [pipe(prompt, generator = g).images[0]]
        scores.append(calc_scores(prompt, pil_images, processor, model, device))
    print('Score: ', sum(s[0] for s in scores)/len(scores))
    wandb.log({'AVG_SCORE': sum(s[0] for s in scores)/len(scores)})

    # Score on N_SMALL for:

    # - CFG scale
    cfg_scales = [2, 4, 6, 8, 10, 12, 14, 16]
    for cfg_scale in cfg_scales:
        scores = []
        for i, prompt in enumerate(prompts[:args.n_small]):
            g = torch.Generator().manual_seed(i)
            pil_images = [pipe(prompt, generator = g, guidance_scale=cfg_scale).images[0]]
            scores.append(calc_scores(prompt, pil_images, processor, model, device))
        wandb.log({
            'score': sum(s[0] for s in scores)/len(scores),
            'cfg_scale': cfg_scale
        })

    # - Steps
    n_steps = [10, 15, 20, 30, 40, 50, 70]
    for n_step in n_steps: 
        scores = []
        for i, prompt in enumerate(prompts[:args.n_small]):
            g = torch.Generator().manual_seed(i)
            pil_images = [pipe(prompt, generator = g, num_inference_steps=n_step).images[0]]
            scores.append(calc_scores(prompt, pil_images, processor, model, device))
        wandb.log({
            'score': sum(s[0] for s in scores)/len(scores),
            'n_steps': n_step
        })

    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description="Eval script")
    parser.add_argument("--n_prompts", type=int, default=200, help="How many prompts to use for testing")
    parser.add_argument("--n_small", type=int, default=10, help="How many prompts to use for testing on shorter tasks")
    parser.add_argument("--pretrained_pipeline_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the pretrained pipeline")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    score(args)
