function [hf, projection_matrix, res_norms] = train_joint(hf, projection_matrix, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, proj_energy, params, init_CG_opts, tau)

% Initial Gauss-Newton optimization of the filter and
% projection matrix.

% Index for the start of the last column of frequencies
lf_ind = cellfun(@(hf) size(hf,1) * (size(hf,2)-1) + 1, hf(1,1,:), 'uniformoutput', false);

% Construct stuff for the proj matrix part
yf_H = cellfun(@(x) permute(x,[3 4 1 2]), yf, 'uniformoutput', false);
% init_samplef_sample_weight = cellfun(@(xf) bsxfun(@times, xf, sample_weights), samplesf, 'uniformoutput', false);

% init_samplef = cellfun(@(x) permute(x, [4 3 1 2]), samplesf, 'uniformoutput', false);
% init_samplef_H = cellfun(@(X) conj(reshape(X, size(X,2), [])), init_samplef, 'uniformoutput', false);


% init_samplef_H = cellfun(@(X) conj(reshape(X, size(X,2), [])), samplesf, 'uniformoutput', false);

% init_samplef_H_samp_weight = cellfun(@(X) conj(reshape(X, size(X,2), [])), init_samplef_samp_weight, 'uniformoutput', false);

% Construct preconditioner
diag_M = cell(size(hf));
diag_M(1,1,:) = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3)) + params.precond_reg_param*reg_energy, sample_energy, reg_energy, 'uniformoutput',false);
diag_M(2,1,:) = cellfun(@(m) params.precond_proj_param * (m + params.projection_reg), proj_energy, 'uniformoutput',false);

% Allocate
rhs_samplef = cell(size(hf));
res_norms = [];

for iter = 1:params.init_GN_iter
    % Project sample with new matrix
    init_samplef_proj = cellfun(@(x,P) mtimesx(x, P, 'speed'), samplesf, projection_matrix, 'uniformoutput', false);
    init_hf = cellfun(@(x) permute(x, [3 4 1 2]), hf(1,1,:), 'uniformoutput', false);
    
    % Construct the right hand side vector for the filter part    
    rhs_samplef(1,1,:) = cellfun(@(xf, yf) bsxfun(@times, conj(permute(xf,[3 4 1 2])), yf), init_samplef_proj, yf, 'uniformoutput', false);
    rhs_samplef(1,1,:) = cellfun(@(x) bsxfun(@times, x, permute(sample_weights,[3 4 1 2])), rhs_samplef(1,1,:), 'uniformoutput', false);
    rhs_samplef(1,1,:) = cellfun(@(x) tau * squeeze(sum(x,3)),rhs_samplef(1,1,:), 'uniformoutput', false);
    
    % Construct the right hand side vector for the projection matrix part
    init_samplef_sample_weight = cellfun(@(xf) bsxfun(@times, xf, sample_weights), samplesf, 'uniformoutput', false);
    xyf = cellfun(@(x,y) bsxfun(@times,x,y),init_samplef_sample_weight, yf_H, 'uniformoutput', false);
    xyf = cellfun(@(x) reshape(sum(x,1), size(x,2),[]), xyf,'uniformoutput', false);
    hf_H = cellfun(@(x) reshape(x, [],size(x,3)), hf(1,1,:), 'uniformoutput', false);
    rhs_samplef(2,1,:) = cellfun(@(P, xyf, hf_H, fi) (tau * 2 *real(xyf * hf_H - xyf(:,fi:end) * hf_H(fi:end,:)) - params.projection_reg * P), ...
        projection_matrix, xyf, hf_H, lf_ind, 'uniformoutput', false);
        
    % Initialize the projection matrix increment to zero
    hf(2,1,:) = cellfun(@(P) zeros(size(P), 'single'), projection_matrix, 'uniformoutput', false);
    
    % do conjugate gradient
    [hf, res_norms_temp] = pcg_ccot(...
        @(x) lhs_operation_joint(x, init_samplef_proj, sample_weights, reg_filter, samplesf, init_hf, params.projection_reg, tau),...
        rhs_samplef, init_CG_opts, ...
        @(x) diag_precond(x, diag_M), ...
        [], @inner_product_joint, hf);
    
    % Make the filter symmetric (avoid roundoff errors)
    hf(1,1,:) = symmetrize_filter(hf(1,1,:));
    
    % Add to the projection matrix
    projection_matrix = cellfun(@plus, projection_matrix, hf(2,1,:), 'uniformoutput', false);
    
    res_norms = [res_norms; res_norms_temp];
end

% Extract filter
hf = hf(1,1,:);

res_norms = res_norms/sqrt(inner_product_joint(rhs_samplef,rhs_samplef));