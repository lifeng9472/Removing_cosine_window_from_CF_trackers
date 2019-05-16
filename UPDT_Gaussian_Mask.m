function results = UPDT_Gaussian_Mask(seq, res_path, bSaveImage, parameters)


if nargin < 4
	parameters = [];
    parameters.debug_level = 0;
end


% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 10;
hog_params.feat_name = 'hog';
hog_params.use_aug = false;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.compressed_dim = 3;
cn_params.feat_name = 'cn';
cn_params.use_aug = false;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.compressed_dim = 3;
ic_params.feat_name = 'ic';
ic_params.use_aug = false;

cnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; % Name of the network
cnn_params.nn_type = 'dagnn';   
cnn_params.output_layer = {'res4f_relu'};               % Which layers to use
cnn_params.downsample_factor = [1];           % How much to downsample each output layer
cnn_params.compressed_dim = [96];
cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)

cnn_params.use_aug = true;
cnn_params.dropout_mask = {'use_random'};

cnn_params.feat_name = {'net_deep'};


% Which features to include
params.t_features = {
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

params.filter_ids = {{'hog', 'cn', 'ic'}, {'net_deep'}};
params.filter_output_sigma_factor = {1/16, 1/4};
params.learning_rate = {0.025,0.0075};	 	 	

params.use_for_scale_estimation = [true, true];

params.min_filter_score = 0.5;

params.prior_alpha_t = [0.5, 0.5];

params.max_num_candidates = 20;

params.neighbor_dist = 0.05;
params.fusion_lamdba_t = 0.15;	 	 	


params.delta_function = 'exp';
params.delta_params = {1};	 	 	

params.output_sz_score = [250,250];

shallow_reg_params.reg_window_min = 1e-4;			% The minimum value of the regularization window
shallow_reg_params.reg_window_edge = 1e-1;         % The impact of the spatial regularization
shallow_reg_params.reg_sparsity_threshold = 0.05;
shallow_reg_params.reg_type = 'normal';

deep_reg_params.reg_window_min = 10e-4;			% The minimum value of the regularization window
deep_reg_params.reg_window_edge = 50e-3;         % The impact of the spatial regularization
deep_reg_params.reg_sparsity_threshold = 0.1;
deep_reg_params.reg_type = 'normal';

params.filter_reg_params = {shallow_reg_params, deep_reg_params};

params.sample_merge_type = {'Merge', 'Merge'};
% Global feature parameters1s
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.5;         % The scaling of the target size to get the search area
params.min_image_sample_size = 200^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

init_frames = struct();
params.init_frames = init_frames;
params.use_multiple_gt_frames = false;

aug_params = struct();
aug_params(1).type = 'original';
aug_params(2).type = 'fliplr';
aug_params(2).param = [];
aug_params(3).type = 'rot';
aug_params(3).param = {5, -5, 10, -10, 20, -20, 30, -30, 45,-45, -60, 60};
aug_params(4).type = 'blur';
aug_params(4).param = {[2, 0.2 ], [0.2, 2], [3,1], [1, 3], [2, 2]};
aug_params(5).type = 'shift';
aug_params(5).param = {[8, 8], [-8, 8 ], [8, -8], [-8,-8]};
aug_params(6).type = 'dropout';
aug_params(6).param = {1,2,3,4, 5, 6, 7};

params.data_aug_params = aug_params;
params.use_data_augmentation = true;

params.use_gt_for_learning = false;
params.use_gt_for_scale = false;

% Learning parameters
params.nSamples = 50;                   % Maximum number of stored training samples
params.sample_replace_strategy = 'lowest_prior';    % Which sample to replace when the memory is full
params.lt_size = 0;                     % The size of the long-term memory (where all samples have equal weight)
params.train_gap = 5;                   % The number of intermediate frames with no training (0 corresponds to training every frame)
params.skip_after_frame = 1;            % After which frame number the sparse update scheme should start (1 is directly)
params.use_detection_sample = false;     % Use the sample that was extracted at the detection stage also for learning

% Factorized convolution parameters
params.use_projection_matrix = true;    % Use projection matrix, i.e. use the factorized convolution formulation
params.update_projection_matrix = true; % Whether the projection matrix should be optimized or not
params.proj_init_method = 'pca';        % Method for initializing the projection matrix
params.projection_reg = 5e-8;

% Generative sample space model parameters
params.use_sample_merge = true;                 % Use the generative sample space model to merge samples
params.distance_matrix_update_type = 'exact';   % Strategy for updating the distance matrix

% Conjugate Gradient parameters
params.CG_iter = 5;                     % The number of Conjugate Gradient iterations in each update after the first frame
params.init_CG_iter = 10*15;            % The total number of Conjugate Gradient iterations used in the first frame
params.init_GN_iter = 10;               % The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
params.CG_use_FR = false;               % Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
params.CG_standard_alpha = true;        % Use the standard formula for computing the step length in Conjugate Gradient
params.CG_forgetting_rate = 75;
params.precond_data_param = 0.3;
params.precond_reg_param = 0.02;
params.precond_proj_param = 35;

% Regularization window parameters
params.use_reg_window = true;           % Use spatial regularization or not
params.reg_window_power = 2;            % The degree of the polynomial to use (e.g. 2 is a quadratic window)

% Interpolation parameters
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel
params.interpolation_centering = true;      % Center the kernel at the feature sample
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel

% Scale parameters for the translation model
% Only used if: params.use_scale_filter = false
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.02;               % The scale factor

% Scale filter parameters
% Only used if: params.use_scale_filter = true
params.use_scale_filter = false;          % Use the fDSST scale filter or not (for speed)

% Visualization
params.visualization = 0;               % Visualiza tracking and detection scores
params.debug = parameters.debug_level;                       % Do full debug visualization

params.save_data = false;

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% Added parameters for optimization in the first frame
params.init_tau = 2.5;  
params.init_scale_factor = 1;
params.init_max_iterations = 5;

% Added parameters for optimization in subsequent frames
params.tau = 2.8;
params.scale_factor = 1;
params.max_iterations = 5;

params.M_delta = 4;  % The hyper-parameter in Eqn. (5)
% Run tracker
results = tracker(params);
