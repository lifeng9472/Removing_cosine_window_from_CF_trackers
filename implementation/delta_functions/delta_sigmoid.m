function d = delta_sigmoid(x, y, id, a, b)
    
% Find dist
d = [x(id), y(id)] - [x', y'];
d = sqrt(d(:,1).^2 + d(:,2).^2);

ids = d < a;

d(ids) =  b*d(ids)/a;
d(~ids) = b;

end

