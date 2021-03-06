# VideoSLIC

> OpenCV implementation of SLIC algorithm applied to video sequences.

[![Codacy](https://app.codacy.com/project/badge/Grade/d780d653e89e48ed96ed2009f2e02deb)](https://www.codacy.com/gh/ClaudiuGeorgiu/VideoSLIC)

This project contains the source code used to obtain the results described in the paper ["Optimizing Superpixel Clustering for Real-Time Egocentric-Vision Applications"](https://ieeexplore.ieee.org/document/6920066). The main purpose of the project is to apply the [SLIC](http://ivrg.epfl.ch/research/superpixels) algorithm to video sequences by using some optimization techniques that aim at obtaining close to real-time performaces. The source code is written in `C++`, is based on the previous work made [here](http://github.com/PSMM/SLIC-Superpixels) and uses the following libraries:
* OpenCV
* Intel TBB (used for parallel execution of the code)
* Boost



## License

You are free to use this code under the [MIT License](https://github.com/ClaudiuGeorgiu/VideoSLIC/blob/master/LICENSE).
