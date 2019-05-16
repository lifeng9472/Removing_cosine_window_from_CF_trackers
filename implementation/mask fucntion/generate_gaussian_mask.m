function M = generate_gaussian_mask(use_sz, base_target_sz, featureRatio, params)
% Generate Gaussian Shaped mask function

% Params initialization
base_target_filter_sz = base_target_sz/featureRatio;
output_sigma_factor_M = params.M_delta;

% Find the regions without discontinuous boundaries
M_scale = use_sz - base_target_sz/featureRatio;
M_range = zeros(numel(M_scale), 2);
for iWH = 1:numel(M_scale)
    if (mod(floor(M_scale(iWH)), 2))
        M_range(iWH,:) = [-floor((M_scale(iWH)-1)/2), floor((M_scale(iWH)-1)/2)];
    else
        M_range(iWH,:) = [-floor(M_scale(iWH)/2), floor(M_scale(iWH)/2)];
    end
end
center = floor((use_sz + 1) / 2) + mod(use_sz + 1, 2);
M_h = (center(1)+ M_range(1,1)) : (center(1) + M_range(1,2));
M_w = (center(2)+ M_range(2,1)) : (center(2) + M_range(2,2));


% Compute the Gaussian shaped mask function
rg           = [-floor((numel(M_h)-1)/2):ceil((numel(M_h)-1)/2)];
cg           = [-floor((numel(M_w)-1)/2):ceil((numel(M_w)-1)/2)];
[rs, cs]     = ndgrid(rg,cg);
gaussian_window  = exp(-0.5 * (rs.^2 / (output_sigma_factor_M * base_target_filter_sz(1))^2 + cs.^2 / (output_sigma_factor_M * base_target_filter_sz(2))^2));
pad_size = use_sz - size(gaussian_window);
pad_size = floor(pad_size / 2); 
gaussian_window = padarray(gaussian_window,pad_size);
gaussian_window = padarray(gaussian_window,[size(gaussian_window,1)~= use_sz(1), 0],'post');
gaussian_window = padarray(gaussian_window,[0, size(gaussian_window,2)~= use_sz(2)],'post');

M = gaussian_window;
M = circshift(M, [-floor((use_sz(1)-1)/2) -floor((use_sz(2)-1)/2)]);