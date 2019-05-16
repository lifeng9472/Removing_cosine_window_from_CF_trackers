function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
% params.nSamples = min(params.nSamples, seq.num_frames);

if ~isfield(params, 'use_gt_for_learning')
    params.use_gt_for_learning = false;
end

if ~isfield(params, 'use_gt_for_scale')
    params.use_gt_for_scale = false;
end

if ~isfield(params, 'output_sz_score')
    params.output_sz_score = [250, 250];
end

if ~isfield(params, 'save_data')
    params.save_data = false;
end

if params.use_gt_for_learning
    params.use_detection_sample = false;
    params.use_gt_for_scale = false;
    if ~isfield(seq, 'ground_truth_rect')
        error('Set pass_gt option in run expt');
    else
        gt_rect_list = seq.ground_truth_rect;
    end
    seq.scores = cell(seq.num_frames,1);
    seq.scores{1} = [];
    seq.score_pos = cell(seq.num_frames,1);
    seq.score_pos{1} = [];
    seq.score_scale = cell(seq.num_frames,1);
    seq.score_scale{1} = [];
end

if params.use_gt_for_scale
    params.use_detection_sample = false;
end

if isfield(seq, 'ground_truth_rect')
    gt_rect_list = seq.ground_truth_rect;
end

if ~isfield(params, 'use_multiple_gt_frames') || ~params.use_multiple_gt_frames
    init_frames_for_training = 1;
else
    if ~isfield(seq, 'ground_truth_rect')
        error('Set pass_gt option in run expt');
    end
    
    % seq_id = find(strcmp({params.init_frames.name},seq.name(1:end-2)),1);
    %if isempty(seq_id)
    %    init_frames_for_training = 1;
    %else
    seq_len = seq.endFrame - seq.startFrame + 1;
    step_len = floor(seq_len / (params.num_init_frames-1));
    % init_frames_for_training = seq.startFrame:step_len:seq.endFrame;
    init_frames_for_training = 1:step_len:seq_len;
    % init_frames_for_training = params.init_frames;
    %end
end


if ~isfield(params, 'use_simplified_fusion')
    params.use_simplified_fusion  = true;
end


if params.use_simplified_fusion
    if any(strcmpi(params.delta_function, {'test1'}))
        params.del_func = @(x,y,z) delta_t1(x,y,z, params.delta_params{:});
    elseif any(strcmpi(params.delta_function, {'smooth'}))
        params.del_func = @(x,y,z) delta_smooth(x,y,z, params.delta_params{:});
    elseif any(strcmpi(params.delta_function, {'exp'}))
        params.del_func = @(x,y,z) delta_exp(x,y,z, params.delta_params{:});
    end
end

if ~isfield(params, 'use_data_augmentation')
    params.use_data_augmentation = false;
end


if params.use_data_augmentation
    if ~strcmp(params.data_aug_params(1).type, 'original')
        error('First sample has to be original')
    end
    
    data_aug_params = struct();
    ct = 1;
    for i=1:length(params.data_aug_params)
        if length(params.data_aug_params(i).param) > 1
            for j=1:length(params.data_aug_params(i).param)
                data_aug_params(ct).type = params.data_aug_params(i).type;
                data_aug_params(ct).param = params.data_aug_params(i).param{j};
                if isfield(params.data_aug_params(1), 'feat_weight')
                    data_aug_params(ct).feat_weight = params.data_aug_params(i).feat_weight{j};
                end
                ct = ct + 1;
            end
        else
            data_aug_params(ct).type = params.data_aug_params(i).type;
            data_aug_params(ct).param = params.data_aug_params(i).param;
            if isfield(params.data_aug_params(1),'feat_weight')
                data_aug_params(ct).feat_weight = params.data_aug_params(i).feat_weight{1};
            end
            ct = ct + 1;
        end
    end
end

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
% try
%     [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
% catch err
%     warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
%     params.use_mexResize = false;
%     global_fparams.use_mexResize = false;
% end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features_dagnn(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');



% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);

use_feat_for_aug = get_feature_data_aug_info(features, num_feature_blocks);

feature_names  = cell(1,1,length(feature_info.dim));
dropout_mask_names = cell(1,1,length(feature_info.dim));
ct = 1;
for i=1:numel(features)
    if iscell(features{i}.fparams.feat_name)
        num_elems_ = length(features{i}.fparams.feat_name);
        feature_names(ct:ct + num_elems_ - 1) = features{i}.fparams.feat_name;
        if isfield(features{i}.fparams, 'dropout_mask')
            dropout_mask_names(ct:ct + num_elems_ - 1) = features{i}.fparams.dropout_mask;
        end
        
        ct = ct + num_elems_;
    else
        feature_names{ct} = features{i}.fparams.feat_name;
        if isfield(features{i}.fparams, 'dropout_mask')
            dropout_mask_names{ct} = features{i}.fparams.dropout_mask;
        end
        ct = ct + 1;
    end
end

% Find which features belong to which filters
filter_ids = cell(1,1,numel(params.filter_ids));
feature_to_filter_map = zeros(num_feature_blocks, 1);

for i=1:numel(filter_ids)
    for j=1:numel(params.filter_ids{i})
        filter_ids{i} = [filter_ids{i}, find(strcmp(params.filter_ids{i}{j}, feature_names))];
    end
end


% TODO check if this works fine
if params.use_data_augmentation
    dropout_mask_all = cell(1,1,numel(dropout_mask_names));
    for f_id = 1:numel(dropout_mask_names)
        if ~isempty(dropout_mask_names{f_id})
            if strcmp(dropout_mask_names{f_id}, 'use_random')
                dropout_mask_all{f_id} = true(feature_dim(f_id), 10);
                
                for i=1:10
                    off_ids = datasample(1:feature_dim(f_id),round(0.2*feature_dim(f_id)),'Replace', false);
                    dropout_mask_all{f_id}(off_ids,i) = false;
                end
            else
                dropout_mask_tmp = load(dropout_mask_names{f_id});
                dropout_mask_all{f_id} = dropout_mask_tmp.dropout_mask;
            end
        else
            % TODO check the number of dropout samples needed
            dropout_mask_all{f_id} = true(feature_dim(f_id), 10);
        end
    end
end


for i=1:num_feature_blocks
    tmp_matches = cellfun(@(a) any(ismember(a,i)), filter_ids, 'uniformOutput',false);
    feature_to_filter_map(i) = find([tmp_matches{:}]);
end

feature_to_filter_map_cell = reshape(num2cell(feature_to_filter_map),1 ,1, []);

% Build use_for_scale
use_for_scale_estimation = params.use_for_scale_estimation;
use_for_scale_estimation_feat = true(numel(feature_to_filter_map), 1);
% for i=1:numel(features)
%     features{i}.fparams.use_for_scale = use_for_scale_estimation(feature_to_filter_map(i));
%     use_for_scale_estimation_feat(i) = use_for_scale_estimation(feature_to_filter_map(i));
% end

% Get feature specific parameters
feature_params = init_feature_params(features, feature_info);
feature_extract_info = get_feature_extract_info(features);

% Set the sample feature dimension
if params.use_projection_matrix
    sample_dim = feature_params.compressed_dim;
else
    sample_dim = feature_dim;
end

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);

sigma_all = cellfun(@(x) params.filter_output_sigma_factor{x}, feature_to_filter_map_cell, 'uniformoutput', false );
% sigma_all = reshape(sigma_all,1,1,[]);

% construct the Gaussian label function using Poisson formula
sig_y = cellfun( @(sig) sqrt(prod(floor(base_target_sz))) * sig * (output_sz ./ img_support_sz), sigma_all , 'uniformoutput', false);
yf_y = cellfun(@(ky, sig) single(sqrt(2*pi) * sig(1) / output_sz(1) * exp(-2 * (pi * sig(1) * ky / output_sz(1)).^2)), ky, sig_y, 'uniformoutput', false);
yf_x = cellfun(@(kx, sig) single(sqrt(2*pi) * sig(2) / output_sz(2) * exp(-2 * (pi * sig(2) * kx / output_sz(2)).^2)), kx, sig_y,'uniformoutput', false);
yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);

yf_full = full_fourier_coeff(yf);
y_full =  cellfun(@(x) real(cifft2(x)), yf_full, 'uniformoutput', false);

% construct cosine window
% cos_window = cellfun(@(sz) tukeywin(sz(1)+2, 0.5)*tukeywin(sz(2)+2,0.5)', feature_sz_cell, 'uniformoutput', false);
% cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
% cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);


cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);


% Compute Fourier series of interpolation function
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

if ~isfield(params, 'use_featurewise_regularization')
    params.use_featurewise_regularization = false;
end

if params.use_featurewise_regularization
    reg_filter = cellfun(@(feat) get_reg_filter(img_support_sz, base_target_sz, params, feat.fparams.reg), reshape(features,1,1,3), 'uniformoutput', false);
else
    reg_filter = cellfun(@(feat_map) get_reg_filter(img_support_sz, base_target_sz, params, params.filter_reg_params{feat_map}), feature_to_filter_map_cell, 'uniformoutput', false); 
end

% Construct spatial regularization filter
% reg_filter = cellfun(@(feat_map) get_reg_filter(img_support_sz, base_target_sz, params, params.filter_reg_params{feat_map}), feature_to_filter_map_cell, 'uniformoutput', false); 


% Compute the energy of the filter (used for preconditioner)
reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% Set conjugate gradient uptions
init_CG_opts.CG_use_FR = true;
init_CG_opts.tol = 1e-6;
init_CG_opts.CG_standard_alpha = true;
init_CG_opts.debug = params.debug;
CG_opts.CG_use_FR = params.CG_use_FR;
CG_opts.tol = 1e-6;
CG_opts.CG_standard_alpha = params.CG_standard_alpha;
CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf % || params.learning_rate >= 1
    CG_opts.init_forget_factor = 0;
else
    % TODO hack?
    CG_opts.init_forget_factor = (1-params.learning_rate{1})^params.CG_forgetting_rate;
end

seq.time = 0;

% Initialize and allocate
prior_weights = cellfun(@(x) zeros(params.nSamples,1, 'single'), params.filter_ids, 'uniformOutput', false);
sample_weights = cellfun(@(x) cast(x, 'like', params.data_type), prior_weights, 'uniformOutput', false);
samplesf = cell(1, 1, num_feature_blocks);


if params.use_data_augmentation
    prior_weights_init = cellfun(@(x) zeros(params.nSamples,1, 'single'), params.filter_ids, 'uniformOutput', false);
    
    for i=1:numel(prior_weights_init)
        if isfield(params.data_aug_params(1), 'feat_weight')
            for samp_id = 1:numel(data_aug_params)
                prior_weights_init{i}(samp_id) = data_aug_params(samp_id).feat_weight(i);
            end
            prior_weights_init{i} = prior_weights_init{i}/ sum(prior_weights_init{i});
        else
            prior_weights_init{i}(1:numel(data_aug_params)) = 1 / numel(data_aug_params);
        end
    end
else
    prior_weights_init = cellfun(@(x) zeros(params.nSamples,1, 'single'), params.filter_ids, 'uniformOutput', false);
    
    for i=1:numel(prior_weights_init)
        prior_weights_init{i}(1) = 1;
    end
end


merged_sample = cell(1,1,num_feature_blocks);
new_sample = cell(1,1,num_feature_blocks);
merged_sample_id = cell(1,1,numel(filter_ids));
new_sample_id = cell(1,1,numel(filter_ids));

if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    % dont have to use mtimesx.
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,sample_dim(k),params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

% Distance matrix stores the square of the euclidean distance between each pair of
% samples. Initialise it to inf
distance_matrix = cellfun(@(x) inf(params.nSamples, 'single'), params.filter_ids, 'uniformOutput', false);

% Kernel matrix, used to update distance matrix
gram_matrix = cellfun(@(x) inf(params.nSamples, 'single'), params.filter_ids, 'uniformOutput', false);

latest_ind = [];
frames_since_last_train = inf(numel(filter_ids),1);
num_training_samples = 0;

if numel(params.skip_after_frame) ~= numel(filter_ids)
    if numel(params.skip_after_frame) ~= 1
        error('bad')
    end
    params.skip_after_frame = params.skip_after_frame*ones(numel(filter_ids),1);
end

if numel(params.train_gap) ~= numel(filter_ids)
    if numel(params.train_gap) ~= 1
        error('bad')
    end
    params.train_gap = params.train_gap*ones(numel(filter_ids),1);
end

% if params.do_merge_fix
params.minimum_sample_weight = cellfun(@(x) x*(1-x)^(2*params.nSamples), params.learning_rate, 'UniformOutput',false);
% else
%    params.minimum_sample_weight = cellfun(@(x) params.learning_rate{1}*(1-params.learning_rate{1})^(2*params.nSamples), params.learning_rate, 'UniformOutput',false);
% send

res_norms = cell(1,1,numel(filter_ids));
residuals_pcg = [];


if params.save_data
    % activation_val = cell(1,1,seq.num_frames);
end

if isnan(seq.num_frames) || isinf(seq.num_frames) 
    seq.num_frames = 10000;
end

fusion_weights = zeros(seq.num_frames, 2);

quad_options = optimoptions('quadprog','Display','off','MaxIterations', 50, 'OptimalityTolerance', 1e-3, 'StepTolerance', 1e-3, 'ConstraintTolerance', 1e-2);

% Define the Gaussian shaped mask functions
M{1} = generate_gaussian_mask([size(yf{k1},1),size(yf{k1},1)], base_target_sz, feature_info.min_cell_size(k1), params);
M{2} = generate_gaussian_mask([size(yf{3},1),size(yf{3},1)], base_target_sz, feature_info.min_cell_size(3), params);

hf_final = cell(1,1,numel(block_inds) + 1);
projection_matrix_final = cell(1,1,numel(block_inds) + 1);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            
            if params.use_gt_for_learning
                gt_rect = gt_rect_list(seq.frame-1, :);
                pos = gt_rect([2, 1]) + (gt_rect([4, 3]) - 1)/2;
                currentScaleFactor = sqrt((gt_rect(3)*gt_rect(4)) / (base_target_sz(1)*base_target_sz(2)));
            end
            
            if params.use_gt_for_scale
                gt_rect = gt_rect_list(seq.frame-1, :);
                currentScaleFactor = sqrt((gt_rect(3)*gt_rect(4)) / (base_target_sz(1)*base_target_sz(2)));
            end
            
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Project sample
            xt_proj = project_sample(xt, projection_matrix);
            
            % Do windowing of features
            xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier seriescos_window
            xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
            end
            
            scores_fs_sum_per_filter = cell(1,1,numel(filter_ids));
            for f_id = 1:numel(filter_ids)
                scores_fs_sum_per_filter{f_id} = zeros(size(scores_fs_feat{k1}), 'like',scores_fs_feat{k1});
            end
            
            
            for k = 1:num_feature_blocks
                filter_id_ = feature_to_filter_map(k);
                scores_fs_sum_per_filter{filter_id_}(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                    scores_fs_sum_per_filter{filter_id_}(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                    scores_fs_feat{k};
            end
            
            % TODO parametrize this
            output_sz_score = params.output_sz_score;
            
            scores_all_blocks = zeros(output_sz_score(1),output_sz_score(2),nScales,numel(filter_ids));
            
            % Sample scores at different scales
            for i=1:numel(scaleFactors)
                for f_id = 1:numel(filter_ids)
                    if use_for_scale_estimation(f_id)
                        samp_sz = round(output_sz_score*scaleFactors(i)/min(scaleFactors));
                        tmp_ = sample_fs(scores_fs_sum_per_filter{f_id}(:,:,:,i), samp_sz);
                    else
                        samp_sz = round(output_sz_score*1/min(scaleFactors));
                        tmp_ = sample_fs(scores_fs_sum_per_filter{f_id}(:,:,:,1), samp_sz);
                    end
                    
                    tmp_ = fftshift(fftshift(tmp_,1),2);
                    r_ = floor((size(tmp_,1) - output_sz_score(1))/2);
                    scores_all_blocks(:,:,i,f_id) = tmp_(1+r_:r_+output_sz_score(1),1+r_:r_+output_sz_score(2));
                end
            end
            %
            %             soln_all_scales = zeros(nScales,3);
            %             f_val_all_scales = zeros(nScales,1);
            %             candidates_all_scales = zeros(nScales,2);
            %
            
            if params.use_simplified_fusion
                
                candidates = [];
                neighbour_candidates = [];
                
                % Find the peaks for each filter
                for f_id = 1:numel(filter_ids)
                    if use_for_scale_estimation(f_id)
                        thresh_mask = scores_all_blocks(:,:,:,f_id) > max(reshape(scores_all_blocks(:,:,:,f_id),[],1))*params.min_filter_score;
                        tmp_map = scores_all_blocks(:,:,:,f_id).*single(thresh_mask);
                        local_peaks = imregionalmax(tmp_map,6);
                        new_candidates = find(local_peaks);
                        
                        if ~isempty(new_candidates)
                            candidates = [candidates; new_candidates(:)];
                        end
                    end
                end
                
                
                % Find peaks for the sum
                scores_joint = zeros(output_sz_score(1),output_sz_score(2), nScales);
                
                for f_id = 1:numel(filter_ids)
                    %if use_for_scale_estimation(f_id)
                    scores_joint = scores_joint + params.prior_alpha_t(f_id)*scores_all_blocks(:,:,:,f_id);
                    %else
                    %   scores_joint = scores_joint + params.prior_alpha_t(f_id)*scores_all_blocks(:,:,1,f_id);
                    %end
                end
                
                thresh_mask = scores_joint > max(reshape(scores_joint,[],1))*params.min_filter_score;
                tmp_map = scores_joint.*single(thresh_mask);
                local_peaks = imregionalmax(tmp_map,6);
                new_candidates = find(local_peaks);
                
                if ~isempty(new_candidates)
                    candidates = [candidates; new_candidates(:)];
                end
                
                scores_size = size(scores_joint);
                
                % Remove if too many candidates
                candidates = unique(candidates);
                candidates_prior_score = scores_joint(candidates);
                
                if numel(candidates_prior_score) > params.max_num_candidates
                    % Sort
                    [~, sorted_ids] = sort(candidates_prior_score, 'descend');
                    candidates = candidates(sorted_ids(1:params.max_num_candidates));
                end
                
                % Find candidates in the neighbourhood
                [candidate_r,candidate_c, candidate_s] = ind2sub(scores_size,candidates);
                
                if isfield(params, 'use_many_neighbours') && params.use_many_neighbours
                    candidate_shift = round(scores_size(1)*params.neighbor_dist);
                    for cand_id = 1:length(candidate_r)
                        new_candidates_n = [candidate_r(cand_id)+candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+3*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-3*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+3*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-3*candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+5*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-5*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+5*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-5*candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+7.5*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-7.5*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+7.5*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-7.5*candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+10*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-10*candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+10*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-10*candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+3*candidate_shift, candidate_c(cand_id)-3*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)-3*candidate_shift, candidate_c(cand_id)+3*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)+3*candidate_shift, candidate_c(cand_id)+3*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)-3*candidate_shift, candidate_c(cand_id)-3*candidate_shift, candidate_s(cand_id);...
                            candidate_r(cand_id)+7*candidate_shift, candidate_c(cand_id)-7*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)-7*candidate_shift, candidate_c(cand_id)+7*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)+7*candidate_shift, candidate_c(cand_id)+7*candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id)-7*candidate_shift, candidate_c(cand_id)-7*candidate_shift, candidate_s(cand_id);...
                            ];
                        
                        new_candidates_n = round(new_candidates_n);
                        out_of_frame = new_candidates_n(:,1) < 1 | new_candidates_n(:,2) < 1 | new_candidates_n(:,1) > (scores_size(1)) | new_candidates_n(:,2) > (scores_size(1));
                        new_candidates_n(out_of_frame,:) = [];
                        
                        
                        new_candidates = sub2ind(scores_size, new_candidates_n(:,1), new_candidates_n(:,2), new_candidates_n(:,3));
                        neighbour_candidates = [neighbour_candidates; new_candidates(:)];
                    end
                    
                else
                    candidate_shift = round(scores_size(1)*params.neighbor_dist);
                    for cand_id = 1:length(candidate_r)
                        new_candidates_n = [candidate_r(cand_id)+candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id)-candidate_shift, candidate_c(cand_id), candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)+candidate_shift, candidate_s(cand_id); ...
                            candidate_r(cand_id), candidate_c(cand_id)-candidate_shift, candidate_s(cand_id)];
                        
                        new_candidates_n = round(new_candidates_n);
                        out_of_frame = new_candidates_n(:,1) < 1 | new_candidates_n(:,2) < 1 | new_candidates_n(:,1) > (scores_size(1)) | new_candidates_n(:,2) > (scores_size(1));
                        new_candidates_n(out_of_frame,:) = [];
                        
                        
                        new_candidates = sub2ind(scores_size, new_candidates_n(:,1), new_candidates_n(:,2), new_candidates_n(:,3));
                        neighbour_candidates = [neighbour_candidates; new_candidates(:)];
                    end
                end
                
                candidates = unique(candidates);
                neighbour_candidates = unique(neighbour_candidates);
                
                % Remove canidates with low peaks
                candidates_prior_score = scores_joint(candidates);
                
                max_cand_score = max(candidates_prior_score);
                
                % TODO parametrize this
                candidates_to_keep = candidates_prior_score > 0.4*max_cand_score;
                candidates = candidates(candidates_to_keep);
                
                n_candidates_prior_score = scores_joint(neighbour_candidates);
                max_n_cand_score = max(n_candidates_prior_score);
                
                % TODO parametrize this
                n_candidates_to_keep = n_candidates_prior_score > 0.4*max_n_cand_score;
                neighbour_candidates = neighbour_candidates(n_candidates_to_keep);
                
                
                scores_all_blocks_cur_scale = reshape(scores_all_blocks, [],2);
                candidates_ys = scores_all_blocks_cur_scale(candidates,1);
                candidates_yd = scores_all_blocks_cur_scale(candidates,2);
                
                n_candidates_ys = scores_all_blocks_cur_scale(neighbour_candidates,1);
                n_candidates_yd = scores_all_blocks_cur_scale(neighbour_candidates,2);
                
                % Find co-ordinates for each candidate
                [candidate_r,candidate_c, candidate_s] = ind2sub(scores_size,candidates);
                [neighbour_candidate_r,neighbour_candidate_c, neighbour_candidate_s] = ind2sub(scores_size,neighbour_candidates);
                
                % Optimize
                alpha_prior_t = params.prior_alpha_t;
                
                soln = cell(1,1,numel(candidates));
                f_val = cell(1,1,numel(candidates));
                
                all_peaks_ys = [candidates_ys;  n_candidates_ys];
                all_peaks_yd = [candidates_yd;  n_candidates_yd];
                all_peaks_r = [candidate_r; neighbour_candidate_r];
                all_peaks_c = [candidate_c; neighbour_candidate_c];
                
                for candidate_id = 1:numel(candidates)
                    % Enforce scale constraint
                    %scale_ys = scores_all_blocks(candidate_r(candidate_id),candidate_c(candidate_id),:,1);
                    
                    %if any(all_peaks_ys(candidate_id) < scale_ys) && params.enforce_scale_margin
                    %    soln{candidate_id} = [];
                    %    f_val{candidate_id} = [];
                    %else
                    H = diag([params.fusion_lamdba_t * 2./alpha_prior_t, 0]);
                    f = [0, 0, -1];
                    
                    A =  [-all_peaks_ys(candidate_id) + all_peaks_ys, -all_peaks_yd(candidate_id) + all_peaks_yd, params.del_func(all_peaks_r/scores_size(1), all_peaks_c/scores_size(2), candidate_id)];
                    
                    b =  zeros(size(A,1),1);
                    Aeq = [1, 1, 0];
                    beq = 1;
                    lb = [0, 0, 0];
                    
                    [soln{candidate_id}, f_val{candidate_id}] = quadprog(H, f, A, b, Aeq, beq, lb, [], [], quad_options);
                    %end
                    
                end
                
                soln_found = cellfun(@(x) ~isempty(x), f_val, 'UniformOutput', false);
                soln_found = cell2mat(soln_found);
                
                soln = soln(soln_found);
                f_val = f_val(soln_found);
                candidate_r = candidate_r(soln_found);
                candidate_c = candidate_c(soln_found);
                candidate_s = candidate_s(soln_found);
                
                % soln_mat = cell2mat(soln);
                f_val_mat = cell2mat(f_val);
                [~, min_id] = min(f_val_mat);
                
                if params.debug > 0
                    fprintf('alpha_sh: %f,  alpha_d: %f,  margin: %f \n', soln{min_id}(1), soln{min_id}(2), soln{min_id}(3));
                    
                    if soln{min_id}(1) < 0.5
                        yo = 1;
                    end
                end
                
                if ~isempty(min_id)
                    fusion_weights(seq.frame, :) = [soln{min_id}(1), soln{min_id}(2)];
                else
                    fusion_weights(seq.frame, :) = [-1, -1];
                end
                
                if isempty(min_id)
                    [max_score, id] = max(scores_joint(:));
                    [trans_row, trans_col, scale_ind] = ind2sub(size(scores_joint), id);
                else
                    trans_row = candidate_r(min_id);
                    trans_col =  candidate_c(min_id);
                    scale_ind = candidate_s(min_id);
                end
                
                
                
            else
                % Find peaks for the sum
                scores_joint = zeros(output_sz_score(1),output_sz_score(2), nScales);
                
                for f_id = 1:numel(filter_ids)
                    %if use_for_scale_estimation(f_id)
                    scores_joint = scores_joint + params.prior_alpha_t(f_id)*scores_all_blocks(:,:,:,f_id);
                    %else
                    %   scores_joint = scores_joint + params.prior_alpha_t(f_id)*scores_all_blocks(:,:,1,f_id);
                    %end
                end
                
                [max_score, id] = max(scores_joint(:));
                [trans_row, trans_col, scale_ind] = ind2sub(size(scores_joint), id);
            end
            
            trans_row = trans_row - (output_sz_score(1)/2) - 1;
            trans_col = trans_col - (output_sz_score(2)/2) - 1;
            
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz_score) * currentScaleFactor * min(scaleFactors);
            scale_change_factor = scaleFactors(scale_ind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            
            
            
            %             if params.debug > 10
            %                 scores_per_channel = bsxfun(@times, hf_full{1}, xtf_proj{1});
            %                 scores_per_channel = fftshift(fftshift(sample_fs(scores_per_channel, [100, 100]),1),2) ;
            %                 figure(32);
            %
            %                 n_dims = size(scores_per_channel,3);
            %                 for bl_id = 1:n_dims
            %                     subplot(n_dims/8,8,bl_id);
            %                     imagesc(scores_per_channel(:,:,bl_id,scale_ind), [0 0.01]);
            %                     axis off
            %                 end
            %             end
            
            
            % Do scale tracking with the scale filter
            if nScales > 0 && params.use_scale_filter
                scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
            end
            
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
        
        %         if params.debug > 10
        %             figure(101);
        %             imagesc(fftshift(sample_fs(scores_fs(:,:,scale_ind))));colorbar; axis image;
        %             % title(sprintf('Scale %i,  max(response) = %f', scale_ind, max(max(response(:,:,scale_ind)))));
        %         end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data Augmentation part
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    using_multiple_init_frames = (params.use_multiple_gt_frames || params.use_data_augmentation);
    num_init_train_samples  = 1;
    
    if seq.frame == 1 && using_multiple_init_frames
        if params.use_multiple_gt_frames && params.use_data_augmentation
            error('Select only one')
        end
        
        if seq.frame == 1 && params.use_data_augmentation
            num_init_train_samples = length(data_aug_params);
            xlf_init_all = cell(1,1,num_init_train_samples);
            xl_init_all = cell(1,1,num_init_train_samples);
        elseif seq.frame == 1 && params.use_multiple_gt_frames
            num_init_train_samples = numel(init_frames_for_training);
            xlf_init_all = cell(1,1,num_init_train_samples);
            xl_init_all = cell(1,1,num_init_train_samples);
        else
            num_init_train_samples = 1;
        end
        
        for init_samp_id=1:num_init_train_samples
            % Load the image and pos
            orignal_pos = pos;
            original_sf = currentScaleFactor;
            im_orig = im;
            
            % Handle multiple init frames
            if ~params.use_multiple_gt_frames && ~params.use_data_augmentation
                im = im_orig;
            elseif params.use_multiple_gt_frames
                if seq.frame == 1
                    seq.frame = init_frames_for_training(init_samp_id) - 1;
                    [~, im] = get_sequence_frame(seq);
                    gt_rect = gt_rect_list(init_frames_for_training(init_samp_id), :);
                    
                    pos = gt_rect([2, 1]) + (gt_rect([4, 3]) - 1)/2;
                    currentScaleFactor = sqrt((gt_rect(3)*gt_rect(4)) / (base_target_sz(1)*base_target_sz(2)));
                    seq.frame = 1;
                else
                    im = im_orig;
                end
            else
                % Handle data aug
                current_aug = data_aug_params(init_samp_id);
                if strcmp(current_aug.type, 'original')
                    im = im_orig;
                elseif strcmp(current_aug.type, 'fliplr')
                    im = fliplr(im_orig);
                    pos(2) = size(im,2) - pos(2) + 1;
                elseif strcmp(current_aug.type, 'occlusion')
                    r1 = round(pos(1) - target_sz(1)*0.5);
                    r2 = r1 + target_sz(1);
                    
                    c1 = round(pos(2) - target_sz(2)*0.5);
                    c2 = c1 + target_sz(2);
                    
                    if current_aug.param == 1
                        r2 = round(r1 + target_sz(1)/3);
                        im(r1:r2,c1:c2,:) = 0;
                    elseif current_aug.param == 2
                        r1 = round(r2 - target_sz(1)/3);
                        im(r1:r2,c1:c2,:) = 0;
                    elseif current_aug.param == 3
                        c2 = round(c1 + target_sz(2)/3);
                        im(r1:r2,c1:c2,:) = 0;
                    elseif current_aug.param == 4
                        c1 = round(c2 - target_sz(2)/3);
                        im(r1:r2,c1:c2,:) = 0;
                    end
                elseif strcmp(current_aug.type, 'dropout')
                    im = im_orig;
                elseif strcmp(current_aug.type, 'shift')
                    pos(1) = pos(1) + current_aug.param(1)*currentScaleFactor;
                    pos(2) = pos(2) + current_aug.param(2)*currentScaleFactor;
                elseif strcmp(current_aug.type, 'noise')
                    cur_noise = abs(25*randn(size(im)));
                    im = im + uint8(cur_noise);
                elseif strcmp(current_aug.type, 'blur')
                    im = imgaussfilt(im, current_aug.param);
                elseif strcmp(current_aug.type, 'bg_mask')
                    % mask the background
                    mask_padding = 0.25*min(target_sz);
                    mask_sz = target_sz + round(mask_padding);
                    mask = zeros(size(im,1), size(im,2));
                    r1 = round(pos(1) - mask_sz(1)*0.5);
                    r2 = r1 + mask_sz(1);
                    
                    c1 = round(pos(2) - mask_sz(2)*0.5);
                    c2 = c1 + mask_sz(2);
                    
                    r1 = max(r1,1);
                    c1 = max(c1,1);
                    r2 = min(r2, size(im,1));
                    c2 = min(c2, size(im,2));
                    
                    mask(r1:r2,c1:c2) = 1;
                    mask = imgaussfilt(mask, mask_padding);
                    
                    im = single(im);
                    im = im.*mask;
                    im = uint8(im);
                elseif strcmp(current_aug.type, 'rot')
                    
                    theta = current_aug.param;
                    
                    
                    T_trans1 = [1, 0, 0; ...
                        0, 1, 0; ...
                        -pos(2), -pos(1), 1];
                    
                    T_rot = [cosd(theta), sind(theta), 0; ...
                        -sind(theta), cosd(theta),  0; ...
                        0, 0, 1];
                    
                    T_trans2 = [1, 0, 0; ...
                        0, 1, 0; ...
                        pos(2), pos(1), 1];
                    
                    tform = affine2d(T_trans1*T_rot*T_trans2);
                    [im, ref] = imwarp(im_orig, tform ) ;
                    
                    % Find transformed location
                    [x1,y1]=transformPointsForward(tform,pos(2),pos(1));
                    
                    pos(2) = round(x1 - ref.XWorldLimits(1));
                    pos(1) = round(y1 - ref.YWorldLimits(1));
                end
            end
            
            
            % Extract sample and init projection matrix
            % Extract image region for training sample
            sample_pos = round(pos);
            sample_scale = currentScaleFactor;
            xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % TODO check if correct
            if params.use_data_augmentation && strcmp(current_aug.type, 'dropout')
                for f_id_t=1:num_feature_blocks
                    dropout_mask = dropout_mask_all{f_id_t}(:,current_aug.param);
                    xl{f_id_t}(:,:,~dropout_mask) = 0;
                    xl{f_id_t} = xl{f_id_t}*size(xl{f_id_t},3) / sum(dropout_mask);
                end
            end
            
            % Remove the cosine window on training stage
            %xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
            xlw = xl;
            
            % Compute the fourier series
            xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
            
            % New sample to be added
            xlf = compact_fourier_coeff(xlf);
            
            % Shift sample
            if params.use_data_augmentation && strcmp(current_aug.type, 'shift')
                shift_samp = 2*pi * (orignal_pos - sample_pos) ./ (sample_scale * img_support_sz);
                xlf = shift_sample(xlf, shift_samp, kx, ky);
            else
                shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
                xlf = shift_sample(xlf, shift_samp, kx, ky);
            end
            
            xlf_init_all{init_samp_id} = xlf;
            xl_init_all{init_samp_id} = xl;
            
            im = im_orig;
            pos = orignal_pos;
            currentScaleFactor = original_sf ;
            
            clear xlw
        end
        
        % Find projection matrix
        if seq.frame == 1
            % Form the proj matrix
            % Form the new data sample
            xl_all_cat = xl_init_all{1};
            
            for tmp_ct_=2:num_init_train_samples
                
                
                for k=1:num_feature_blocks
                    if params.use_data_augmentation
                        if use_feat_for_aug(k)
                            xl_all_cat{k} = cat(1,xl_all_cat{k}, xl_init_all{tmp_ct_}{k});
                        end
                    else
                        xl_all_cat{k} = cat(1,xl_all_cat{k}, xl_init_all{tmp_ct_}{k});
                    end
                end
            end
            
            projection_matrix = init_projection_matrix(xl_all_cat, sample_dim, params);
        end
        
        xlf_all_final = cell(1,1,num_feature_blocks);
        
        if params.update_projection_matrix
            for k = 1:num_feature_blocks
                xlf_all_final{k} = zeros(params.nSamples,feature_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
            end
        else
            for k = 1:num_feature_blocks
                xlf_all_final{k} = zeros(params.nSamples, sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
            end
        end
        
        xlf_proj = project_sample(xlf_init_all{1}, projection_matrix);
        
        for tmp_ct_=1:num_init_train_samples
            % Project sample
            if seq.frame == 1 && ~params.update_projection_matrix
                xlf_init_all{tmp_ct_} = project_sample(xlf_init_all{tmp_ct_}, projection_matrix);
            end
            
            if tmp_ct_ == 1 &&  seq.frame == 1
                % Set xlf proj to first sample
                xlf_proj_orig = xlf_init_all{tmp_ct_};
            end
            
            if params.use_data_augmentation && tmp_ct_ > 1
                xlf_init_all{tmp_ct_}(~use_feat_for_aug) = xlf_proj_orig(~use_feat_for_aug);
            end
            
            % The permuted sample is only needed for the CPU implementation
            for k = 1:num_feature_blocks
                xlf_all_final{k}(tmp_ct_,:,:,:) = permute(xlf_init_all{tmp_ct_}{k}, [4 3 1 2]);
            end
        end
    elseif seq.frame == 1
        % Extract image region for training sample
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % Remove the cosine window on training stage
        xlw = xl;
        %xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
        
        % New sample to be added
        xlf = compact_fourier_coeff(xlf);
        
        % Shift sample
        
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf = shift_sample(xlf, shift_samp, kx, ky);
        
        % Init the projection matrix
        projection_matrix = init_projection_matrix(xl, sample_dim, params);
        
        % Project sample
        xlf_proj = project_sample(xlf, projection_matrix);
        
        xlf_init_all = cell(1,1,1);
        if params.update_projection_matrix
            xlf_all_final = cell(1,1,num_feature_blocks);
            
            for k = 1:num_feature_blocks
                xlf_all_final{k} = zeros(params.nSamples,feature_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
            end
            
            xlf_init_all{1} = xlf;
            for k = 1:num_feature_blocks
                xlf_all_final{k}(1,:,:,:) = permute(xlf_init_all{1}{k}, [4 3 1 2]);
            end
        end
        
        clear xlw
    elseif params.learning_rate{1} > 0
        if ~params.use_detection_sample
            if params.use_gt_for_learning
                tracked_pos = pos;
                tracked_scale_factor = currentScaleFactor;
                gt_rect = gt_rect_list(seq.frame, :);
                pos = [gt_rect(2), gt_rect(1)] + (gt_rect([4, 3]) - 1)/2;
                currentScaleFactor = sqrt((gt_rect(3)*gt_rect(4)) / (base_target_sz(1)*base_target_sz(2)));
            end
            
            if params.use_gt_for_scale
                tracked_scale_factor = currentScaleFactor;
                gt_rect = gt_rect_list(seq.frame-1, :);
                currentScaleFactor = sqrt((gt_rect(3)*gt_rect(4)) / (base_target_sz(1)*base_target_sz(2)));
            end
            
            % Extract image region for training sample
            sample_pos = round(pos);
            sample_scale = currentScaleFactor;
            xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Project sample
            xl_proj = project_sample(xl, projection_matrix);
            
            % Remove the cosine window on training stage
            %xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xlf1_proj = interpolate_dft(xlf1_proj, interp1_fs, interp2_fs);
            
            % New sample to be added
            xlf_proj = compact_fourier_coeff(xlf1_proj);
        else
            if params.debug > 1
                % Only for visualization
                xl = cellfun(@(xt) xt(:,:,:,scale_ind), xt, 'uniformoutput', false);
            end
            
            % Use the sample that was used for detection
            sample_scale = sample_scale(scale_ind);
            
            scale_indices = cell(1,1,numel(xtf_proj));
            scale_indices(use_for_scale_estimation_feat) = {scale_ind};
            scale_indices(~use_for_scale_estimation_feat) = {1};
            
            xlf_proj = cellfun(@(xf, ind) xf(:,1:(size(xf,2)+1)/2,:,ind), xtf_proj,scale_indices, 'uniformoutput', false);
        end
        
        % Shift the sample so that the target is centered
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
        
        if params.use_gt_for_learning
            pos = tracked_pos;
            currentScaleFactor = tracked_scale_factor;
        end
    end
    
    if ~params.use_gpu
        xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    end
    
    
    % Insert sample to memory
    % xlf_proj_all = cell(1,1,numel(xlf_init_all));
    if seq.frame > 1 || seq.frame == 1 && ~using_multiple_init_frames && ~params.update_projection_matrix
        if params.use_sample_merge
            % Update the samplesf to include the new sample. The distance
            % matrix, kernel matrix and prior weight are also updated
            for filt_id=1:numel(filter_ids)
                cur_ids = filter_ids{filt_id};
                
                [merged_sample(cur_ids), new_sample(cur_ids), merged_sample_id{filt_id}, new_sample_id{filt_id}, distance_matrix{filt_id}, gram_matrix{filt_id}, prior_weights{filt_id}] = ...
                    update_sample_space_model(samplesf(cur_ids), xlf_proj_perm(cur_ids), distance_matrix{filt_id}, gram_matrix{filt_id}, prior_weights{filt_id},...
                    num_training_samples,params.learning_rate{filt_id},params, params.sample_merge_type{filt_id}, params.minimum_sample_weight{filt_id});
            end
            
            if num_training_samples < params.nSamples
                num_training_samples = num_training_samples + 1;
            end
        end
        
        % Insert the new training sample
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id{feature_to_filter_map_cell{k}} > 0
                    samplesf{k}(:,:,:,merged_sample_id{feature_to_filter_map_cell{k}}) = merged_sample{k};
                end
                if new_sample_id{feature_to_filter_map_cell{k}} > 0
                    samplesf{k}(:,:,:,new_sample_id{feature_to_filter_map_cell{k}}) = new_sample{k};
                end
            else
                if merged_sample_id{feature_to_filter_map_cell{k}} > 0
                    samplesf{k}(merged_sample_id{feature_to_filter_map_cell{k}},:,:,:) = merged_sample{k};
                end
                if new_sample_id{feature_to_filter_map_cell{k}} > 0
                    samplesf{k}(new_sample_id{feature_to_filter_map_cell{k}},:,:,:) = new_sample{k};
                end
            end
        end
        
        sample_weights = cellfun(@(x) cast(x, 'like', params.data_type), prior_weights, 'uniformOutput', false);
    else
        sample_weights = prior_weights_init;
    end
    
    train_tracker = (seq.frame(:) < params.skip_after_frame(:)) | (frames_since_last_train(:) >= params.train_gap(:));
    
    if any(train_tracker)
        % Used for preconditioning
        new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        
        if seq.frame == 1
            % Initialize stuff for the filter learning
            
            % Initialize Conjugate Gradient parameters
            sample_energy = new_sample_energy;
            CG_state = cell(1,1,numel(filter_ids));
            
            if params.update_projection_matrix
                % Number of CG iterations per GN iteration
                init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
                
                hf = cell(2,1,num_feature_blocks);
                proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
            else
                CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
                
                hf = cell(1,1,num_feature_blocks);
            end
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
            end
        else
            CG_opts.maxit = params.CG_iter;
            
            % Update the approximate average sample energy using the learning
            % rate. This is only used to construct the preconditioner.
            sample_energy = cellfun(@(se, nse, feat_map) (1 - params.learning_rate{feat_map}) * se + params.learning_rate{feat_map} * nse, sample_energy, new_sample_energy, feature_to_filter_map_cell,'uniformoutput', false);
        end
        
        % Do training
        if (seq.frame == 1 && params.update_projection_matrix) || (seq.frame == 1 && using_multiple_init_frames)
            if params.update_projection_matrix
                                
                hf_init = hf;
                projection_matrix_init = projection_matrix;
                % The features need dimension reduction at first.                
                for filt_id = 1:numel(filter_ids)
                    iter = 1;
                    tau = params.init_tau;
                    cur_ids = filter_ids{filt_id};
                    mult_term = tau ./ (M{filt_id} + tau);
                    
                    if(filt_id == 1)
                        max_k1 = k1;
                    else
                        max_k1 = 3;
                    end
                    
                    xlf_final = xlf_all_final(cur_ids);
                    xlf_final_full = cell(1,1,numel(xlf_final));
                    
                    for j = 1:numel(xlf_final)
                        for k = 1:params.nSamples
                            xlf_final_full{j}(:,:,:,k) = full_fourier_coeff(permute(xlf_final{j}(k,:,:,:),[3 4 2 1]));
                        end
                    end
                    
                    % Iteratively update between auxiliary variables z and UPDT model parameters (both filter and projection matrix)

                    while iter <= params.init_max_iterations
                        scores_f_sum = [];
                        if(iter == 1)
                            hf = hf_init(1,:,cur_ids);
                        end
                        
                        hf_full = full_fourier_coeff(hf);
                        
                        
                        % Solve the auxiliary variables z

                        if(iter == 1)
                            projection_matrix = projection_matrix_init(1,1,cur_ids);
                        end
                        
                        cur_samplesf = project_sample(xlf_final_full, projection_matrix);
                        
                        for j = 1:numel(hf_full)
                            for k = 1:params.nSamples                                
                                if(numel(hf_full) == 1)
                                    scores_f_sum(:,:,:,k) = sum(bsxfun(@times, hf_full{1}, cur_samplesf{1}(:,:,:,k)), 3);                                    
                                elseif(j == k1 && (numel(hf_full) > 1))
                                    scores_f_sum(:,:,:,k) = sum(bsxfun(@times, hf_full{k1}, cur_samplesf{k1}(:,:,:,k)), 3);                                                                    
                                end
                            end
                        end
                        
                        
                        if(filt_id == 1)
                            for j = block_inds(1)
                                for k = 1:params.nSamples
                                    tmp = sum(bsxfun(@times, hf_full{j}, cur_samplesf{j}(:,:,:,k)), 3);
                                    scores_f_sum(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),1,k) = ...
                                        scores_f_sum(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),1,k) + tmp;
                                end
                            end                            
                        end
                        
                        z = zeros(size(mult_term,1), size(mult_term,2),params.nSamples);
                        
                        res_term = zeros(size(y_full{max_k1},1),size(y_full{max_k1},2),size(scores_f_sum,4));
                        
                        for k = 1:size(scores_f_sum,4)
                            res_term(:,:,k) = squeeze(cifft2(scores_f_sum(:,:,:,k))) - y_full{max_k1};
                        end
                        
                        z(:,:,1:params.nSamples) = bsxfun(@times, res_term, mult_term);


                        % Solve the UPDT model parameters (both the filters and projection matrix)
                        
                        yzf{1,1,max_k1} = cfft2(bsxfun(@plus, z, y_full{max_k1}));
                        
                        tmp = bsxfun(@plus, z, y_full{max_k1});
                        for k = 1:size(scores_f_sum,4)
                            yzf{1,1,max_k1}(:,:,k) = cfft2(tmp(:,:,k));
                        end
                        
                        if(filt_id == 1)
                            for j = block_inds(1)
                                yzf{1,1,j} = yzf{1,1,k1}(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),:);
                            end
                        end
                        
                        
                        yzf(cur_ids) = compact_fourier_coeff(yzf(cur_ids));
                        
                        % Do Conjugate gradient optimization of both the filter and projection matrix
                        [hf, projection_matrix, res_norms{filt_id}] = train_joint(hf, projection_matrix, xlf_all_final(cur_ids), yzf(cur_ids), reg_filter(cur_ids), sample_weights{filt_id}, sample_energy(cur_ids), reg_energy(cur_ids), proj_energy(cur_ids), params, init_CG_opts, tau);
                        
                        tau = tau * params.init_scale_factor;
                        
                        iter = iter + 1;
                    end
                    
                    hf_final(:,:,filter_ids{filt_id}) = hf;
                    projection_matrix_final(:,:,filter_ids{filt_id}) = projection_matrix;
                end            
            end
            
            % get the final filters for the first frame
            hf = hf_final;
            projection_matrix = projection_matrix_final;
            
            
            for tmp_id_=1:num_init_train_samples
                xlf_init_all{tmp_id_} = project_sample(xlf_init_all{tmp_id_}, projection_matrix);
            end
            
            % Insert all samples
            % Re-project and insert training sample
            for tmp_id_=1:num_init_train_samples
                % xlf_proj = project_sample(xlf_init_all{tmp_id_}, projection_matrix);
                xlf_proj_perm = cellfun(@(x) permute(x, [4 3 1 2]), xlf_init_all{tmp_id_}, 'uniformOutput',false);

                for filt_id=1:numel(filter_ids)
                    cur_ids = filter_ids{filt_id};
                    
                    [merged_sample(cur_ids), new_sample(cur_ids), merged_sample_id{filt_id}, new_sample_id{filt_id}, distance_matrix{filt_id}, gram_matrix{filt_id}, prior_weights{filt_id}] = ...
                        update_sample_space_model(samplesf(cur_ids), xlf_proj_perm(cur_ids), distance_matrix{filt_id}, gram_matrix{filt_id}, prior_weights{filt_id},...
                        num_training_samples,params.learning_rate{filt_id}, params,params.sample_merge_type{filt_id}, params.minimum_sample_weight{filt_id});
                end
                
                
                if num_training_samples < params.nSamples
                    num_training_samples = num_training_samples + 1;
                end
                
                for k = 1:num_feature_blocks
                    if merged_sample_id{feature_to_filter_map_cell{k}} > 0
                        samplesf{k}(merged_sample_id{feature_to_filter_map_cell{k}},:,:,:) = merged_sample{k};
                    end
                    if new_sample_id{feature_to_filter_map_cell{k}} > 0
                        samplesf{k}(new_sample_id{feature_to_filter_map_cell{k}},:,:,:) = new_sample{k};
                    end
                end
            end
            
            prior_weights = prior_weights_init;
        else
            % Do Conjugate gradient optimization of the filter
            for filt_id=1:numel(filter_ids)
                if ~train_tracker(filt_id)
                    continue;
                end
                scores_f_sum = [];
                cur_ids = filter_ids{filt_id};
                cur_samplesf = [];
                
                for j = 1:numel(cur_ids)
                    for k = 1:params.nSamples
                        cur_samplesf{j}(:,:,:,k) = full_fourier_coeff(permute(samplesf{cur_ids(j)}(k,:,:,:),[3 4 2 1]));
                    end
                end
                
                
                if(filt_id == 1)
                    max_k1 = k1;
                else
                    max_k1 = 3;
                end
                
                iter = 1;
                tau = params.tau;

                % Iteratively update between auxiliary variables z and UPDT model parameters (only the filter)
                
                while iter <= params.max_iterations
                    
                    % Subproblem z
                    hf_full = full_fourier_coeff(hf(cur_ids));
                    mult_term = tau ./ (M{filt_id} + tau);                                     
                    
                    % Solve the auxiliary variables z
                    
                    for j = 1:numel(hf_full)
                        for k = 1:params.nSamples
                            if(numel(hf_full) == 1)
                                scores_f_sum(:,:,:,k) = sum(bsxfun(@times, hf_full{1}, cur_samplesf{1}(:,:,:,k)), 3);
                            elseif(j == k1 && (numel(hf_full) > 1))
                                scores_f_sum(:,:,:,k) = sum(bsxfun(@times, hf_full{k1}, cur_samplesf{k1}(:,:,:,k)), 3);
                            end
                        end
                    end
                    
                    
                    if(filt_id == 1)
                        for j = block_inds(1)
                            for k = 1:params.nSamples
                                tmp = sum(bsxfun(@times, hf_full{j}, cur_samplesf{j}(:,:,:,k)), 3);
                                scores_f_sum(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),1,k) = ...
                                    scores_f_sum(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),1,k) + tmp;
                            end
                        end
                    end


                    z = zeros(size(mult_term,1), size(mult_term,2),params.nSamples);
                    
                    res_term = zeros(size(y_full{max_k1},1),size(y_full{max_k1},2),size(scores_f_sum,4));
                    for k = 1:size(scores_f_sum,4)
                        res_term(:,:,k) = squeeze(cifft2(scores_f_sum(:,:,:,k))) - y_full{max_k1};
                    end
                    
                    z(:,:,1:params.nSamples) = bsxfun(@times, res_term, mult_term);


                    % Solve the UPDT model parameters (only the filters)

                    yzf{1,1,max_k1} = cfft2(bsxfun(@plus, z, y_full{max_k1}));
                    
                    tmp = bsxfun(@plus, z, y_full{max_k1});
                    for k = 1:size(scores_f_sum,4)
                        yzf{1,1,max_k1}(:,:,k) = cfft2(tmp(:,:,k));
                    end
                    
                    if(filt_id == 1)
                        for j = block_inds(1)
                            yzf{1,1,j} = yzf{1,1,k1}(1+pad_sz{j}(1):end-pad_sz{j}(1), 1+pad_sz{j}(2):end-pad_sz{j}(2),:);
                        end
                    end                    
                    
                    yzf(cur_ids) = compact_fourier_coeff(yzf(cur_ids));

                    
                    % Do Conjugate gradient optimization of the filter
                    [hf(cur_ids), res_norms{filt_id}, CG_state{filt_id}] = train_filter(hf(cur_ids), samplesf(cur_ids), yzf(cur_ids), reg_filter(cur_ids), sample_weights{filt_id}, sample_energy(cur_ids), reg_energy(cur_ids), params, CG_opts, CG_state{filt_id}, tau);
                    
                   tau = tau * params.scale_factor;
                   iter = iter + 1;
                end                                                        
            end
        end
        
        % Reconstruct the full Fourier series
        hf_full = full_fourier_coeff(hf);
        
        frames_since_last_train(train_tracker) = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end
    
    % Update the scale filter
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    if params.save_data
        if seq.frame == 1
            init_filter = hf_full;
        end
        
        activation_val{seq.frame} = cell(1,1,num_feature_blocks);
        for ct=1:num_feature_blocks
            tmp_sz = size(xl{ct});
            center_val = round((tmp_sz+1)/2);
            activation_val{seq.frame}{ct} = xl{ct}(center_val(1)-1:center_val(1)+1, center_val(2)-1:center_val(2)+1,:);
        end
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    
    % Hack to save scores, and where they are extracted
    seq = report_tracking_result(seq, tracking_result);
    if seq.frame > 1
        seq.scores{seq.frame} = scores_fs_sum_per_filter;
        seq.score_pos{seq.frame} = sample_pos;
        seq.score_scale{seq.frame} = sample_scale;
    end
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % debug visualization
%     if params.debug > 2
%         figure(20)
%         %         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
%         subplot_cols = num_feature_blocks;
%         subplot_rows = 3;%ceil(feature_dim/subplot_cols);
%         for disp_layer = 1:num_feature_blocks;
%             subplot(subplot_rows,subplot_cols,disp_layer);
%             imagesc(mean(abs(sample_fs(conj(hf_full{disp_layer}))), 3));
%             colorbar;
%             axis image;
%             subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
%             imagesc(mean(abs(xl{disp_layer}), 3));
%             colorbar;
%             axis image;
%             if seq.frame > 1
%                 subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
%                 imagesc(fftshift(sample_fs(scores_fs_feat{disp_layer}(:,:,1,scale_ind))));
%                 colorbar;
%                 axis image;
%             end
%         end
%     end
%     
%     if params.debug > 30
%         if train_tracker
%             residuals_pcg = [residuals_pcg; res_norms];
%             res_start_ind = max(1, length(residuals_pcg)-300);
%             figure(99);plot(res_start_ind:length(residuals_pcg), residuals_pcg(res_start_ind:end));
%             axis([res_start_ind, length(residuals_pcg), 0, min(max(residuals_pcg(res_start_ind:end)), 0.2)]);
%         end
%     end
    
    % visualization
%     if params.visualization == 1
%         rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
%         im_to_show = double(im)/255;
%         if size(im_to_show,3) == 1
%             im_to_show = repmat(im_to_show, [1 1 3]);
%         end
%         if seq.frame == 1,  %first frame, create GUI
%             fig_handle = figure('Name', 'Tracking');
%             %             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
%             imagesc(im_to_show);
%             hold on;
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
%             hold off;
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%             
%             %             output_name = 'Video_name';
%             %             opengl software;
%             %             writer = VideoWriter(output_name, 'MPEG-4');
%             %             writer.FrameRate = 5;
%             %             open(writer);
%         else
%             % Do visualization of the sampled confidence scores overlayed
%             resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
%             
%             % To visualize the continuous scores, sample them 10 times more
%             % dense than output_sz.
%             %sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));
%             
%             figure(fig_handle);
%             %                 set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
%             imagesc(im_to_show);
%             hold on;
%             %resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
%             %alpha(resp_handle, 0.5);
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
%             hold off;
%             
%             %                 axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%         end
%         
%         drawnow
%         %         if frame > 1
%         %             if frame < inf
%         %                 writeVideo(writer, getframe(gcf));
%         %             else
%         %                 close(writer);
%         %             end
%         %         end
%         %          pause
%     end
end

% close(writer);

[seq, results] = get_sequence_results(seq);

if params.save_data
    results.activations = activation_val;
    results.init_filter = init_filter;
    results.final_filter = hf_full;
    % results.projection_matrix = projection_matrix;
    results.scores = seq.scores;
    results.score_pos = seq.score_pos;
    results.score_scale = seq.score_scale;
    
    results.fusion_weights = fusion_weights;
    
end


disp(['fps: ' num2str(results.fps)])
