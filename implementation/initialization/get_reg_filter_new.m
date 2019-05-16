function reg_filter = get_reg_filter_new(sz, target_sz, params, reg_params)

% Compute the spatial regularization function and derive the corresponding
% filter operation used for the optimization

if params.use_reg_window
   num_coefficients = (reg_params.num_coeff+1) / 2;
   
   multi_factor = 1:1:(num_coefficients-1);
   
   theta1 = pi*target_sz.*reg_params.x1./sz;
   theta2 = pi*target_sz.*reg_params.x2./sz;
   
   A = [1 2*cos(theta1(1)*multi_factor) 2*cos(theta1(2)*multi_factor); ...
        1 2*cos(theta2(1)*multi_factor) 2*cos(theta2(2)*multi_factor)];
   
   b = [reg_params.c1; reg_params.c2];
   
   H = A'*A;
   ft = -b'*A;
   
   Aeq = [1, 2*ones(1,2*num_coefficients-2)];
   beq = reg_params.c0;
   
   theta01r = 0:(theta1(1)/3):theta1(1);
   theta01c = 0:(theta1(2)/3):theta1(2);
   
   theta2er = theta2(1):(pi-theta2(1))/3:pi;
   theta2ec = theta2(2):(pi-theta2(2))/3:pi;
   theta2er = theta2er(1:end);
   theta2ec = theta2ec(1:end);
   
   num_theta01 = numel(theta01r); 
   num_theta2e = numel(theta2er); 
   
   Aueq = [ones(num_theta01,1) 2*cos(theta01r'.*multi_factor) 2*ones(num_theta01,num_coefficients-1); ...
           ones(num_theta01,1) 2*ones(num_theta01,num_coefficients-1) 2*cos(theta01c'.*multi_factor) ; ...
           ones(num_theta2e,1) 2*cos(theta2er'.*multi_factor) 2*ones(num_theta2e,num_coefficients-1);...
           ones(num_theta2e,1) 2*ones(num_theta2e,num_coefficients-1) 2*cos(theta2ec'.*multi_factor)];
   
   Aueq = -Aueq;
   bueq = [reg_params.c0*ones(2*num_theta01,1); ...
           reg_params.c2*ones(2*num_theta2e,1)];
   
   bueq = -bueq;
   
   x = quadprog(H,ft,Aueq,bueq,Aeq,beq);
   x = x';
   reg_filter = zeros(reg_params.num_coeff, reg_params.num_coeff);
   reg_filter(num_coefficients,num_coefficients:end) = x(1:num_coefficients);
   reg_filter(num_coefficients,1:num_coefficients-1) = fliplr(x(2:num_coefficients));
   reg_filter(1:num_coefficients-1, num_coefficients) = fliplr(x(num_coefficients+1:end));
   reg_filter(num_coefficients+1:end, num_coefficients) = x(num_coefficients+1:end);
   
else
    % else use a scaled identity matrix
    reg_filter = cast(params.reg_window_min, 'like', params.data_type);
end







