function d_out = delta_smooth(x, y, id, a, b)
    
% Find dist
d = [x(id), y(id)] - [x(:), y(:)];
d = sqrt(d(:,1).^2 + d(:,2).^2);

d_out = d;
% Region1
ids = d <= a;
d_out(ids) = 0.5*(d(ids)/a).^b;


% Region2
ids = (a <  d) & (d < 2*a);

d_out(ids) =  1 - 0.5*((2*a - d(ids))/a).^b;

ids = d >= 2*a;
d_out(ids) = 1;

end

