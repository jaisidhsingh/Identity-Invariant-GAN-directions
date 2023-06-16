import torch
from PIL import Image
import os
import random
import argparse
from tqdm import tqdm
from train import *


def get_concat_h(im1, im2):
	dist = Image.new('RGB', (im1.width + im2.width, im1.height))
	dist.paste(im1, (0, 0))
	dist.paste(im2, (im1.width, 0))
	return dist

def main(args):
	data = torch.load(f"./results/idd-no-mse/all_directions.pt").cuda()

	G = get_generator()
	G = freeze_model(G)

	psi_value = random.random()
	class_labels = None
	z_0 = torch.randn([1, G.z_dim]).to(device)
	
	w_0 = G.mapping(z_0, class_labels, truncation_psi=psi_value, truncation_cutoff=8)
	w_0.requires_grad = False

	base_img = G.synthesis(w_0, noise_mode='const', force_fp32=True)
	base_img = (base_img.permute(0, 2, 3, 1)*127.5+128).clamp(0, 255).to(torch.uint8)
	base_img = base_img[0].cpu().numpy()
	base_img = Image.fromarray(base_img, 'RGB')

	for step_size in tqdm(args.step_sizes):
		w = w_0 + step_size*(data / data.norm(2))

		img = G.synthesis(w, noise_mode='const', force_fp32=True)
		img = (img.permute(0, 2, 3, 1)*127.5+128).clamp(0, 255).to(torch.uint8)
		img = img[0].cpu().numpy()
		img = Image.fromarray(img, 'RGB')

		base_img = get_concat_h(base_img, img)

	save_folder = os.path.join(args.save_folder, args.run_name)
	os.makedirs(save_folder, exist_ok=True)
		
	base_img.save(os.path.join(save_folder, f"result_{args.experiment_id}.png"))

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-name", type=str, default="./final_figures")
	parser.add_argument("--experiment-id", type=int, default=0)
	parser.add_argument("--step-sizes", type=list, default=[0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100])
	parser.add_argument("--save-folder", type=str, default="./inference/multistep")
	
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()

	for i in range(4):
		args.experiment_id = i
		main(args)