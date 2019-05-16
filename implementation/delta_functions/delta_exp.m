function d_out = delta_exp(x, y, id, a)
    
% Find dist
d = [x(id), y(id)] - [x(:), y(:)];
d = sqrt(d(:,1).^2 + d(:,2).^2);

d_out = 1-exp(-(d/(a/9)).^2);

end

