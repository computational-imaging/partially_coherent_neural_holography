import torch
import torch.nn as nn
import utils.utils as utils
import numpy as np
import time
from propagation_ASM import propagation_ASM
from spectrum import wvl2transmission_measured
from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame
import random

class PartialProp(nn.Module):
    """Propagates a SLM phase with multiple wavelengths and sum at the target plane in field

    Class initialization parameters
    -------------------------------
    :param distance: propagation distance in m.
    :param feature_size: feature size of SLM.
    :param wavelength_central: principal wavelength of spectrum.
    :param num_wvls: number of wavelengths.
    :param sample_wavelength_rate: sampling rate between wavelengths
                                   (if evenly sampled, or not, use it for determining wavelength range of sampling)
    :param source_diameter: diameter of aperture of LED (only for LED)
    :param f_col: focal length of the first collimating lens after the source
    :param proptype: 'ASM'
    :param randomly_sampled: boolean, if True, randomly sample angles/wvs every forward pass.
    :param use_sampling_pool: boolean, if True, use pre-defined sampling pool for angle and wvs.
    :param src_type: source type, default 'LED', could be 'sLED'.
    :param batch_size: number of samples every prop
    :param num_angs: number of angles (if using sampling pool)
    :param source_amp_sigma: sigma value used for modeling source amplitude
    :param use_kernel: If True, model partially spatial coherence as a simple kernel.
    Usage
    -----
    Functions as a pytorch module:

    >>> multi_propagate = PartialProp(...)
    >>> output_field = multi_propagate(slm_phase)

    slm_phase: encoded phase-only representation at SLM plane , with dimensions
        [batch, 1, height, width]
    output_field: output_field at the target plane, with dimensions [batch,
        1, height, width, 2]
    """

    def __init__(self, distance=0.1, feature_size=6.4e-6, wavelength_central=532e-9, linear_conv=True,
                 num_wvls=1, sample_wavelength_rate=1e-9, source_diameter=100e-6, f_col=200e-3, image_res=(1080, 1920),
                 proptype='ASM', randomly_sampled=True, use_sampling_pool=True, src_type='LED',
                 device=torch.device('cpu'), batch_size=1, num_angs=1, source_amp_sigma=10e-6,
                 use_kernel=False, initial_kernel='point', initial_kernel_size=None, slm_noise=0.0, fwhm=None,
                 ):

        super(PartialProp, self).__init__()

        if slm_noise > 0.0:  # Simple noise model used in Fig. S6
            torch.manual_seed(0)
            mean_noise = torch.zeros(1, 1, *image_res).to(device)
            std_noise = slm_noise * torch.ones(1, 1, *image_res).to(device)
            self.slm_noise = torch.normal(mean_noise, std_noise)
        else:
            self.slm_noise = None
        
        self.prop_dist = distance
        self.batch_size = batch_size
        self.feature_size = (feature_size
                             if hasattr(feature_size, '__len__')
                             else [feature_size] * 2)
        self.image_res = image_res
        self.precomped_H = None
        self.precomped_H_exp = None
        self.linear_conv = linear_conv
        self.sample_wv_rate = sample_wavelength_rate
        self.randomly_sample = randomly_sampled    # False: backpropagate through fixed wv/spatial thing

        self.use_sampling_pool = use_sampling_pool # True: ramdomly pick wavelengths and spatial distributions
                                                   #       at pinhole at the very first and pick over the pool
                                                   # False: every iteration pick new wvs / tilted waves randomly

        # middle wavelength value
        self.wv_center = wavelength_central

        # maximum incident angle from the edge of the source pinhole (in angular frequency, rad)
        self.w_max = 2 * np.pi / wavelength_central * source_diameter / 2 / f_col
        self.sigma_w = 2 * np.pi / wavelength_central * source_amp_sigma / f_col

        # uniformly sample wvls for initial
        self.src_type = src_type
        self.fwhm = fwhm
        self.pick_wvs(wavelength_central, 0, sample_wavelength_rate, num_wvls, src_type=src_type, device=device)
        self.wv_delta = self.num_wvls / 2 * self.sample_wv_rate

        ##################################
        # modeling low-spatial-coherence #
        ##################################
        assert not (use_kernel and (num_angs > 1))
        if use_kernel:
            self.low_spatial_coherence = self.spatial_kernels(initial_kernel, initial_kernel_size,
                                                              num_wvls, wavelength_central,
                                                              distance, f_col, feature_size,
                                                              source_diameter, image_res,
                                                              device)
            self.source_amp_angular = [1.]
            self.source_phase = 0.
        else:
            self.low_spatial_coherence = None
            if src_type == 'sLED':
                source_amp = [1.] * 1
                self.source_amp_angular = source_amp
                self.source_phase = 0.
            elif src_type == 'LED':
                source_amp = []
                if not randomly_sampled:
                    # 1) Amplitude: Manually sample and keep these samples during iterations
                    ws = manual_angles(num_angs, self.w_max)
                    for wx, wy in ws:
                        source_amp.append(np.exp(-(wx**2 + wy**2) / (2.0 * self.sigma_w**2)))
                    self.source_amp_angular = source_amp

                    # 2) Phase: Render field from the angles, then extract phase
                    source_field = self.source_field(ws, feature_size, image_res).to(device).detach()
                    source_field.requires_grad = False
                    _, self.source_phase = utils.rect_to_polar(source_field[..., 0], source_field[..., 1])
                else:  # random sampling
                    # If use sampling pool : make sampling pool and pick from there.
                    # otherwise : do nothing and just pick randomly every iteration.
                    if use_sampling_pool:
                        # 1) Amp: Assume gaussian shape intensity over the pinhole
                        r = self.w_max * np.random.random(num_angs)
                        theta = 2 * np.pi * np.random.random(num_angs)
                        self.ws = np.array([(wx, wy) for wx, wy in zip(r * np.cos(theta), r * np.sin(theta))])
                        for wx, wy in self.ws:
                            source_amp.append(np.exp(-(wx**2 + wy**2) / (2.0 * 2.0 * self.sigma_w**2)))
                        self.source_amp_angular = np.array(source_amp)

                        # 2) Phase
                        source_field = self.source_field(self.ws, feature_size, image_res).to(device).detach()
                        source_field.requires_grad = False
                        _, self.source_phase = utils.rect_to_polar(source_field[..., 0], source_field[..., 1])

        if proptype == 'ASM':
            self.prop = propagation_ASM_broadband

        # set a device for initializing the precomputed objects
        try:
            self.dev = device#next(self.parameters()).device
        except StopIteration:  # no parameters
            self.dev = torch.device('cpu')


    def forward(self, phases):
        # 1. precompute the kernels only once
        if self.precomped_H is None:
            self.calculate_Hs()

        # 2. randomly sample the coefficients
        source_amp, source_phase, precomped_H, a_ang, q_wv = self.sample_ang_wvs()

        # consider phases of incident beam from different angles
        processed_phase = phases + source_phase
        if self.slm_noise is not None:
            processed_phase += self.slm_noise

        # propagate from SLM to target plane
        real, imag = utils.polar_to_rect(torch.ones_like(processed_phase), processed_phase)
        processed_complex = torch.stack((real, imag), -1)
        processed_complex = torch.view_as_complex(processed_complex)
        output_complex = self.prop(processed_complex, self.feature_size,
                                   None, self.prop_dist, H=precomped_H,
                                   linear_conv=self.linear_conv)

        if self.low_spatial_coherence is not None:
            # all the amplitudes are converted into intensity, then are applied convolution
            # shift-invariant kernel for low-spatial coherence (finite size of light source)
            # sum over n wavelengths (n channels) -> 1 channel
            intensity = q_wv * (output_complex.abs() ** 2)
            intensity = self.low_spatial_coherence(intensity)
        else:
            # Stochastic sampling
            intensity = (q_wv * a_ang) * (output_complex.abs() ** 2)
            intensity = torch.mean(intensity, dim=1, keepdim=True)
            intensity = torch.sum(intensity, dim=0, keepdim=True)

        # convert back to amplitude (lose phase info)
        output_amp = torch.pow(intensity, 0.5)
        output_field = torch.stack(utils.polar_to_rect(output_amp, torch.zeros_like(output_amp)), -1)
        return torch.view_as_complex(output_field)


    def calculate_Hs(self, verbose=True):
        '''
        calculate and stack propagation kernels in channel dimensions
        '''
        if verbose:
            t0 = time.time()
            print('  - computing: propagation kernels...')

        self.precomped_H = None
        for wavelength in self.wvls:
            if self.precomped_H is None:
                self.precomped_H = propagation_ASM(torch.empty(1, 1, 1080, 1920),
                                    self.feature_size, wavelength,
                                    z=self.prop_dist, return_H=True,
                                    linear_conv=True)
            else:
                # stack wavelengths in channel dimension
                self.precomped_H = torch.cat((self.precomped_H,
                                                propagation_ASM(torch.empty(1, 1, 1080, 1920),
                                                                self.feature_size, wavelength,
                                                                z=self.prop_dist, return_H=True,
                                                                linear_conv=True)), dim=1)
        self.precomped_H = self.precomped_H.detach().to(self.dev)
        self.precomped_H.requires_grad = False

        if verbose:
            print(f'  - done:      propagation kernels... took{time.time() - t0:.4f}s')


    def spatial_kernels(self, initial_kernel, initial_kernel_size, num_wvls,
                              wavelength_central, distance, f_col, feature_size, led_size_source_plane,
                              image_res, device):
        # consider spatial coherence as a simple convolution layer applied for intensity
        # Note that we can do this in frequency domain - see Deng et al. 2017, Park 2020.

        # calculate the kernel size
        if initial_kernel_size is not None:
            kernel_size = initial_kernel_size
        else:
            led_size_recon_plane = [led_size_source_plane * distance / f_col] * 2
            dx_recon_plane = [wavelength_central * distance / (N * dx)
                              for N, dx in zip(image_res, feature_size)]
            kernel_size = [round(s / dx) for s, dx in zip(led_size_recon_plane, dx_recon_plane)]

        # use modified pytorch prototyping from Vincent
        low_spatial_coherence = Conv2dSame(num_wvls, 1, kernel_size=kernel_size, bias=False)

        if initial_kernel == 'point':
            # initial kernel is a central point
            initial_weight = torch.zeros(1, num_wvls, *kernel_size).to(device)
            initial_weight[..., int(kernel_size[0] / 2), int(kernel_size[1] / 2)] = 1.  # functions as floor
            low_spatial_coherence.net[1].weight = nn.Parameter(initial_weight)
        elif initial_kernel == 'uniform':
            # initial kernel is an uniform rectangle
            initial_weight = (torch.ones(1, num_wvls, *kernel_size)
                              / np.prod(kernel_size)).detach().to(device)
            low_spatial_coherence.net[1].weight = nn.Parameter(initial_weight)

        else:
            # designated kernel
            initial_weight = torch.tensor(initial_kernel,
                                            dtype=torch.float32).repeat(1, num_wvls, 1, 1).detach().to(device)

        low_spatial_coherence.net[1].weight = nn.Parameter(initial_weight)

        return low_spatial_coherence


    def source_field(self, angles, feature_size, image_res):
        '''
        Stack several plane waves in minibatch dimension
        :param angles: np 1d array of tuples (wy, wx)
        :param feature_size:
        :param image_res:
        :return:
        '''
        source_field = None
        for w in angles:
            # here this amplitude can be a function of wx and wy!
            wy, wx = w
            amp = 1.

            if source_field is None:
                source_field = tilted_plane_wave(amp=amp, w=(wy, wx),
                                                 feature_size=feature_size,
                                                 field_resolution=image_res)
            else:
                source_field = torch.cat((source_field, tilted_plane_wave(amp=amp, w=(wy, wx),
                                                                         feature_size=feature_size,
                                                                         field_resolution=image_res)
                                         ), dim=1)

        return source_field


    def pick_wvs(self, wv_center, wv_delta=0, sample_wv_rate=1e-9, num_wvls=1, rand=False,
                 src_type='LED', device=torch.device('cuda:0')):
        """
        randomly pick wv_center +- wv_delta or wv_center +- n * sample_wv_rate.

        :param wv_center:
        :param wv_delta:
        :param sample_wv_rate:
        :param num_wvls:
        :param rand:
        :param src_type:
        :param device:
        :return:
        """
        if not rand:
            self.wvls = np.array([wv_center + d * sample_wv_rate
                            for d in range(round((-num_wvls+1)/2), round((num_wvls+1)/2))])
        else:
            self.wvls = np.random.uniform(wv_center - wv_delta,
                                          wv_center + wv_delta, (num_wvls))

        if self.fwhm is not None:
            sigma = self.fwhm / (2 * np.sqrt(2*np.log(2)))
            trans = [np.exp(-(wvl - wv_center)**2 / (2*sigma**2)) for wvl in self.wvls]
        else:
            trans = [wvl2transmission_measured(wvl, src_type) for wvl in self.wvls]
        self.trans = nn.Parameter(torch.tensor([x / max(trans) for x in trans], dtype=torch.float32).
                                              to(device), requires_grad=False) # normalize


    def sample_ang_wvs(self):
        num_angs = len(self.source_amp_angular)
        num_wvs = len(self.wvls)
        if self.randomly_sample:  # randomly sample, every iteration:
            # pick kernels and source_phases randomly and pair them (number of batch size)
            if self.use_sampling_pool:
                # pick wavelengths
                m = random.choices(range(len(self.wvls)), k=self.batch_size)
                precomped_H = self.precomped_H[:, m, ...]

                # pick tilted waves
                if self.src_type == 'LED':
                    n = random.choices(range(len(self.ws)), k=self.batch_size)
                    source_phase = self.source_phase[:, n, ...]
                    source_amp = self.source_amp_angular[n]
                else:
                    source_amp = 1.
                    source_phase = 0.
            else:
                # pick wavelengths
                self.pick_wvs(self.wv_center, self.wv_delta, num_wvls=self.batch_size, rand=True, device=self.dev)
                self.calculate_Hs(verbose=False)
                precomped_H = self.precomped_H

                # pick tilted waves
                if self.src_type == 'LED':
                    # From cartesian
                    ws = []
                    while True:
                        num_pick = self.batch_size - len(ws)
                        if num_pick < 1:
                            break
                        wxs, wys = self.w_max*(2*np.random.random(num_pick)-1), \
                                   self.w_max*(2*np.random.random(num_pick)-1)
                        for wx, wy in zip(wxs, wys):
                            r = np.sqrt(wx**2 + wy**2)

                            # restrict within circle shape aperture
                            if r <= self.w_max:
                                ws.append((wx, wy))

                    source_amp = []
                    for wx, wy in ws:
                        source_amp.append(np.exp(-(wx**2 + wy**2) / (2.0 * 2.0 * self.sigma_w**2)))
                    source_field = self.source_field(ws, self.feature_size, self.image_res).to(self.dev).detach()
                    source_field.requires_grad = False
                    _, source_phase = utils.rect_to_polar(source_field[..., 0], source_field[..., 1])

                elif self.src_type == 'sLED':
                    # No spatial distribution for sLED
                    source_amp = 1.
                    source_phase = 0.
        else:
            # uniformly sampled
            if self.use_sampling_pool:
                if num_wvs > num_angs:
                    m = random.choices(range(num_wvs), k=num_angs)
                    precomped_H = self.precomped_H[:, m, ...]
                    source_phase = self.source_phase
                    source_amp = self.source_amp_angular
                else:
                    n = random.choices(range(num_angs), k=num_wvs)
                    source_phase = self.source_phase[:, n, ...]
                    source_amp = self.source_amp_angular[:, n, ...]
                    precomped_H = self.precomped_H
            else:
                source_phase = self.source_phase
                source_amp = self.source_amp_angular
                precomped_H = self.precomped_H

        # calculate coefficients
        if self.src_type == 'LED':
            # span weights along angles on channel dimension
            a_ang = torch.tensor(source_amp, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.dev)
        elif self.src_type == 'sLED':
            a_ang = torch.tensor(source_amp, dtype=torch.float32).to(self.dev)
        q_wv = self.trans.reshape(1, len(self.trans), 1, 1)
        if self.use_sampling_pool and num_wvs > num_angs:
            q_wv = q_wv[:, m, ...]

        return source_amp, source_phase, precomped_H, a_ang, q_wv


    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)
        if slf.precomped_H_exp is not None:
            slf.precomped_H_exp = slf.precomped_H_exp.to(*args, **kwargs)
        # try setting dev based on some parameter, default to cpu
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf


    @property
    def num_wvls(self):
        return len(self.wvls)


    @property
    def training(self):
        return self._training


    @training.setter
    def training(self, mode):
        if mode:
            self.zernike_eval = None  # reset when switching to training
        self._training = mode


def propagation_ASM_broadband(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', H=None,
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method
    # Assume H are always given

    Inputs
    ------
    u_in: complex field of size (num_images, 1, height, width, 2)
        where the last two channels are real and imaginary values
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width, 2)
    """

    if linear_conv:
        # preprocess with padding for linear conv
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')\

    # convolution of the system
    U2 = H * U1

    # Fourier transform of the convolution to the observation plane
    u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        return utils.crop_image(u_out, input_resolution, stacked_complex=False)
    else:
        return u_out


def tilted_plane_wave(w, amp=1.0,
                      feature_size=(6.4e-6, 6.4e-6), field_resolution=(1080, 1920), dtype=torch.float32):
    '''
    return a complex wave field that comes from a shifted source position at source plane after collimating lens

    :param wavelength:
    :param f_col: the focal length of collimating lens
    :param amp:
    :param pos_src:
    :param feature_size:
    :param field_resolution:
    :param dtype:
    :return:
    '''

    dy, dx = feature_size
    y = np.linspace(-dy * field_resolution[0] / 2,
                     dy * field_resolution[0] / 2,
                     field_resolution[0])
    x = np.linspace(-dx * field_resolution[1] / 2,
                     dx * field_resolution[1] / 2,
                     field_resolution[1])
    X, Y = np.meshgrid(x, y)
    wy, wx = w

    phase = wx*X + wy*Y
    phase = torch.tensor(phase, dtype=dtype, requires_grad=False)
    phase = torch.reshape(phase, (1, 1, phase.size()[0], phase.size()[1]))

    real, img = utils.polar_to_rect(amp * torch.ones_like(phase),phase)
    field = torch.stack((real,img), 4)

    return field


def manual_angles(num_angs, w_max):
    ws = [(0., 0.)]
    if num_angs == 5:
        wx, wy = w_max * np.cos(np.pi / 4), w_max * np.sin(np.pi / 4)
        ws.append((wx, 0.))
        ws.append((-wx, 0.))
        ws.append((0., wy))
        ws.append((0., -wy))
    elif num_angs == 9:
        wx, wy = w_max * np.cos(np.pi / 4), w_max * np.sin(np.pi / 4)
        ws.append((wx, 0.))
        ws.append((-wx, 0.))
        ws.append((0., wy))
        ws.append((0., -wy))
        ws.append((wx, wy))
        ws.append((-wx, wy))
        ws.append((-wx, wy))
        ws.append((wx, -wy))
    return np.array(ws)
