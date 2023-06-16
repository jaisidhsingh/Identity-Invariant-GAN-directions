import torch
import torch.nn as nn
import dnnlib
import pickle
from PIL import Image
from torchvision import transforms as tf
import os
import random
from tqdm import tqdm
import argparse

from matcher_configs.elasticface_config import config as cfg
from matcher_backbones.elasticface_backbone import iresnet100

"""
Algo:
------
	1. make a parameter of dims (18, 512) which will be our id-invariant direction

	2. start off from a known latent w_0, and get w = w_0 + alpha*(d / d.norm(2))
	   where alpha is some distance scalar and d is our parameterized direction.
	
	3. already prepare the face image of w_0 which is I_0 and the matcher embedding E_0.
	
	4. pass w through frozen G and get the face image I.
	
	5. pass I through frozen M and get the embedding E.
	
	6. get loss as MSE(E, E_0) and backprop to learn the direction.

Ideally, this direction should be the same for each image. (seems this way from some testing).
"""

device = "cuda"


class IdInvariantDirection(nn.Module):
	def __init__(self, w_0, w_space_dims=(18, 512)):
		super().__init__()
		self.w_0 = w_0
		self.idd = nn.Parameter(torch.randn(w_space_dims).to(device))
	
	def forward(self, alpha):
		return self.w_0 + alpha*(self.idd / self.idd.norm(2))


def get_matcher():
	matcher_ckpt_path = os.path.join(f"matcher_checkpoints/elasticarcface/295672backbone.pth")
	matcher_model = iresnet100(num_features=cfg.embedding_size).to(device)
	state_dict = torch.load(matcher_ckpt_path)
	matcher_model.load_state_dict(state_dict)
	matcher_model = matcher_model.to(device)
	matcher_model.eval()

	matcher_transform = tf.Compose([
		tf.Resize((112, 112)),
		tf.PILToTensor(),
	])

	return matcher_model, matcher_transform

def get_generator():
	model_path = os.path.join("pretrained_models/ffhq.pkl")
	with open(model_path, "rb") as f:
		G = pickle.load(f)["G_ema"].to(device)
	G.eval()
		
	return G

def freeze_model(model):
	for param in model.parameters():
		param.requires_grad = False
	return model

def main(args):
	G = get_generator()
	G = freeze_model(G)

	matcher, matcher_transforms = get_matcher()
	matcher = freeze_model(matcher)

	all_directions = torch.zeros((args.num_latent_iterations, 18, 512))

	global_steps = 0
	for latent_iteration in range(args.num_latent_iterations):

		# get a random W latent point
		psi_value = random.random()
		class_labels = None
		z_0 = torch.randn([1, G.z_dim]).to(device)
		
		w_0 = G.mapping(z_0, class_labels, truncation_psi=psi_value, truncation_cutoff=8)
		w_0.requires_grad = False

		base_img_tensor = G.synthesis(w_0, noise_mode='const', force_fp32=True)
		base_img = (base_img_tensor.permute(0, 2, 3, 1)*127.5+128).clamp(0, 255).to(torch.uint8)
		base_img = base_img[0].cpu().numpy()
		base_img = Image.fromarray(base_img, 'RGB')

		base_img_for_matcher = matcher_transforms(base_img).unsqueeze(0).float().to(device)
		base_img_feats = matcher(base_img_for_matcher)
		base_img_feats.requires_grad = False
		
		idmodel = IdInvariantDirection(w_0)
		
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(idmodel.parameters(), lr=args.learning_rate)

		alpha = 0

		# setup progress bar
		bar = tqdm(total=args.optimization_iterations)
		bar.set_description(f"Latent no. {latent_iteration+1}")

		# start optimzation
		for opt_iter in range(args.optimization_iterations):

			# take a step in the direction that we want to optimize
			alpha += args.alpha_increment

			# forward pass
			optimizer.zero_grad()
			w = idmodel(alpha)

			img_tensor = G.synthesis(w, noise_mode='const', force_fp32=True)
			img = (img_tensor.permute(0, 2, 3, 1)*127.5+128).clamp(0, 255).to(torch.uint8)
			img = img[0].cpu().numpy()
			img = Image.fromarray(img, 'RGB')

			img_for_matcher = matcher_transforms(img).unsqueeze(0).float().to(device)
			img_feats = matcher(img_for_matcher)
			img_feats.requires_grad = True	
			
			# get loss
			loss = criterion(img_feats, base_img_feats) + args.mse_weight*criterion(img_tensor, base_img_tensor)

			# backward pass
			loss.backward()
			optimizer.step()

			# update logging
			global_steps += 1
			logs = {
				"global_steps": global_steps,
				"loss": loss.item()
			}
			bar.update(1)
			bar.set_postfix(**logs)

		# save all directions
		all_directions[latent_iteration] = idmodel.idd.cpu()
		bar.write(f"Direction optimized latent index: {latent_iteration}")
		bar.write(f"Optimization loss: {loss.item()}")
		bar.write(" ")

	save_dir = os.path.join(args.save_dir, args.run_name)
	os.makedirs(save_dir, exist_ok=True)

	save_path = os.path.join(save_dir, "all_directions.pt")
	torch.save(all_directions, save_path)
	bar.write("Done!")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--num-latent-iterations", type=int, default=1)
	parser.add_argument("--optimization-iterations", type=int, default=2000)
	parser.add_argument("--alpha-increment", type=int, default=1e-1)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--mse-weight", type=float, default=0.0)
	parser.add_argument("--run-name", type=str, default="idd-no-mse")
	parser.add_argument("--save-dir", type=str, default="./results")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	main(args)
