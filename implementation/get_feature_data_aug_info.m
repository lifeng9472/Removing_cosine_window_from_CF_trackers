function use_feat_for_data_aug = get_feature_data_aug_info(features, num_feature_blocks)

use_feat_for_data_aug = zeros(num_feature_blocks,1);

ct = 1;
for i=1:length(features)
    num_features = length(features{i}.fparams.nDim);
    if isfield(features{i}.fparams,'use_aug')
        use_feat_for_data_aug(ct:ct+num_features-1) = features{i}.fparams.use_aug;
    end
    
    ct = ct+num_features;
end

end

