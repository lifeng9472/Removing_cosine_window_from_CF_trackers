function net = load_cnn(fparams, im_size)

addpath(genpath('/media/usr/Work4/dataset/code/vot-2018 toolkit/vot-toolkit/vot-workspace/trackers/UPDT/external_libs/matconvnet23'));

net = dagnn.DagNN.loadobj(load(['/media/usr/Work4/networks/' fparams.nn_name])) ;
output_index = net.getLayerIndex(fparams.output_layer);

total_layer_size = numel(net.layers);
total_layer_names = {net.layers.name};
for i = total_layer_size:-1:(output_index + 1)
    net.removeLayer(total_layer_names{i});
end

if strcmpi(fparams.input_size_mode, 'cnn_default')
    base_input_sz = net.meta.normalization.imageSize(1:2);
elseif strcmpi(fparams.input_size_mode, 'adaptive')
    base_input_sz = im_size(1:2);
else
    error('Unknown input_size_mode');
end

net.meta.normalization.imageSize(1:2) = round(base_input_sz .* fparams.input_size_scale);
net.meta.normalization.averageImageOrig = net.meta.normalization.averageImage;

if isfield(net.meta,'inputSize')
    net.meta.inputSize = base_input_sz;
end

if size(net.meta.normalization.averageImage,1) > 1 || size(net.meta.normalization.averageImage,2) > 1
    net.meta.normalization.averageImage = imresize(single(net.meta.normalization.averageImage), net.meta.normalization.imageSize(1:2));
end

end