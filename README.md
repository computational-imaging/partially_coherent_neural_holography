# Speckle-free Holography with Partially Coherent Light Sources and Camera-in-the-loop Calibration
### [Project Page](https://www.computationalimaging.org/publications/partiallycoherentholography/)  | [Paper](https://www.science.org/doi/10.1126/sciadv.abg5040)

[Yifan Peng*](http://stanford.edu/~evanpeng/), [Suyeon Choi*](https://choisuyeon.github.io/), [Jonghyun Kim](http://j-kim.kr/), [Gordon Wetzstein](http://stanford.edu/~gordonwz/)

<font size="1">* Authors contributed equally.</font>

This repository contains the scripts associated with the Science Advances paper "Speckle-free Holography with Partially Coherent Light Sources and Camera-in-the-loop Calibration"

## Getting Started

First, load the [submodules](https://github.com/computational-imaging/neural-holography) in ```neural_holography``` folder with
```
git submodule init
git submodule update
```

Also, you can modify the spectrum information in spectra folder based on measured spectrum from your own setup.

## High-level structure

The code is organized as follows:

* ```main.py``` generates phase patterns with our partially coherent propagatator via SGD/CITL
* ```propagation_partial.py``` contains the partially coherent wave propagation operator implementation.
* ```spectrum.py``` contains utility functions for reading measured spectra.

./neural-holography/: See [here](https://github.com/computational-imaging/neural-holography) for descriptions.


## Running the test
The SLM phase patterns can be reproduced with

SGD with the partially coherent model:
```
python main.py --channel=0 --method=SGD --prop_model=model --root_path=./phases
```

SGD with Camera-in-the-loop optimization:
```
python main.py --channel=0 --method=SGD --prop_model=model --citl=True --root_path=./phases
```


## Citation
If you find our work useful in your research, please cite:

```
@article{Peng:2021:PartiallyCoherent,
author = {Yifan Peng  and Suyeon Choi  and Jonghyun Kim  and Gordon Wetzstein },
title = {Speckle-free holography with partially coherent light sources and camera-in-the-loop calibration},
journal = {Science Advances},
volume = {7},
number = {46},
pages = {eabg5040},
year = {2021},
doi = {10.1126/sciadv.abg5040}
```

## License
This project is licensed under the following license, with exception of the file "data/1.png", which is licensed under the [CC-BY](https://creativecommons.org/licenses/by/3.0/) license.


Copyright (c) 2021, Stanford University

All rights reserved.

Redistribution and use in source and binary forms for academic and other non-commercial purposes with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code, including modified source code, must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form or a modified form of the source code must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of The Leland Stanford Junior University, any of its trademarks, the names of its employees, nor contributors to the source code may be used to endorse or promote products derived from this software without specific prior written permission.

* Where a modified version of the source code is redistributed publicly in source or binary forms, the modified source code must be published in a freely accessible manner, or otherwise redistributed at no charge to anyone requesting a copy of the modified source code, subject to the same terms as this agreement.

THIS SOFTWARE IS PROVIDED BY THE TRUSTEES OF THE LELAND STANFORD JUNIOR UNIVERSITY "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE LELAND STANFORD JUNIOR UNIVERSITY OR ITS TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contact
If you have any questions, please contact

* Yifan (Evan) Peng, evanpeng@stanford.edu
* Suyeon Choi, suyeon@stanford.edu
* Gordon Wetzstein, gordon.wetzstein@stanford.edu
