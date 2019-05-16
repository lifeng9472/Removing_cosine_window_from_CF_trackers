# Remove Cosine Window from Correlation Filter-based Visual Trackers: When and How

# Abstract 
Correlation filters (CFs) have been continuously advancing the state-of-the-art tracking performance and have been extensively studied in the recent few years. Most of the existing CF trackers adopt a cosine window to spatially reweight base image to alleviate boundary discontinuity. However, cosine window emphasizes more on the central region of base image and has the risk of contaminating negative training samples during model learning. On the other hand, spatial regularization
deployed in many recent CF trackers plays a similar role as cosine window by enforcing spatial penalty on CF coefficients. Therefore, we in this paper investigate the feasibility to remove cosine window from CF trackers with spatial regularization. When simply removing cosine window, CF with spatial regularization
still suffers from small degree of boundary discontinuity. To tackle this issue, binary and Gaussian shaped mask functions are further introduced for eliminating boundary discontinuity while reweighting the estimation error of each training sample, and can be incorporated with multiple CF trackers with spatial
regularization. In comparison to the counterparts with cosine window, our methods are effective in handling boundary discontinuity and sample contamination, thereby benefiting tracking performance. Extensive experiments on three benchmarks show that our methods perform favorably against the state-of-the-art trackers using either handcrafted or deep CNN features.

# Installation

1. Clone the GIT repository:

   $ git clone https://github.com/lifeng9472/Removing_cosine_window_from_CF_trackers.git

2. Clone the submodules.  
   In the repository directory, run the commands:

   $ git submodule init  
   $ git submodule update

3. Start Matlab and navigate to the repository.  
   Run the install script:

   |>> install


**Note**

1. This package requires matconvnet [1] and PDollar Toolbox [2]. Both these externals should be installed under the path `./external_libs/`.

2. To run the tracker, you need to change the location of the pre-trained CNN with `absolute path` rather than the `relative path` in feature_extraction/load_CNN.m.
   In addition, `imagenet-resnet-50-dag.mat` is also required and can be downloaded at http://www.vlfeat.org/matconvnet/pretrained/.

## References

[1] Webpage: http://www.vlfeat.org/matconvnet/  
    GitHub: https://github.com/vlfeat/matconvnet

[2] Piotr Dollár.  
    "Piotr’s Image and Video Matlab Toolbox (PMT)."  
    Webpage: https://pdollar.github.io/toolbox/  
    GitHub: https://github.com/pdollar/toolbox  
