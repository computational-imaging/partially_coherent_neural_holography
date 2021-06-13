"""
Speckle-free Holography with Partially Coherent Light Sources and Camera-in-the-loop Calibration:

This is the main executive script used for the phase optimization using SGD + camera-in-the-loop (CITL).

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:PartiallyCoherent,
author = {Y. Peng, S. Choi, J. Kim, G. Wetzstein},
title = {Speckle-free Holography with Partially Coherent Light Sources and Camera-in-the-loop Calibration},
journal = {Science Advances},
year = {2021},
}

-----

$ python main.py --channel=0 --algorithm=SGD --root_path=./phases
"""

import os
import sys
sys.path.append('neural-holography')
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.augmented_image_loader import ImageLoader
from utils.modules import SGD, PhysicalProp
from propagation_ASM import propagation_ASM
from propagation_partial import PartialProp

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./neural-holography/data', help='Directory for the dataset')
p.add_argument('--src_type', type=str, default='sLED', help='sLED or LED')
p.add_argument('--citl', type=utils.str2bool, default=False, help='Use of Camera-in-the-loop optimization with SGD')
p.add_argument('--experiment', type=str, default='', help='Name of experiment')
p.add_argument('--lr', type=float, default=6e-3, help='Learning rate for phase variables (for SGD)')
p.add_argument('--lr_s', type=float, default=1e-3, help='Learning rate for learnable scale (for SGD)')
p.add_argument('--num_iters', type=int, default=1000, help='Number of iterations (SGD)')

# parse arguments
opt = p.parse_args()
run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format
if opt.citl:
    run_id = f'{run_id}_citl'

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')
if opt.citl:
    print(f'    - with camera-in-the-loop ...')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (10 * cm, 10 * cm, 10 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (634.8 * nm, 510 * nm, 450 * nm)[channel]   # SLED
if opt.src_type == 'LED':
    wavelength = (633 * nm, 532 * nm, 460 * nm)[channel]   # LED
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# Hardware setup for CITL
if opt.citl:
    camera_prop = PhysicalProp(channel, laser_arduino=True, roi_res=(roi_res[1], roi_res[0]), slm_settle_time=0.12,
                               range_row=(220, 1000), range_col=(300, 1630),
                               patterns_path=f'F:/citl/calibration',
                               show_preview=True)
else:
    camera_prop = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model

elif opt.prop_model.upper() == 'MODEL':
    propagator = PartialProp(distance=prop_dist, feature_size=feature_size, batch_size=12,
                             wavelength_central=wavelength, num_wvls=15,
                             sample_wavelength_rate=1*nm,
                             randomly_sampled=True,
                             use_sampling_pool=True,
                             f_col=200*mm,
                             source_diameter=75*um,
                             source_amp_sigma=30*um,
                             src_type=opt.src_type, # 'sLED' or 'LED'
                             device=device).to(device)
    propagator.eval()


# Select Phase generation method, algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, opt.citl, camera_prop, writer, device)

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Loop over the dataset
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    # run algorithm (See algorithm_modules.py and algorithms.py)
    # iterative methods, initial phase: random guess
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
    final_phase = phase_only_algorithm(target_amp, init_phase)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')
