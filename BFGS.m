%--------------------------------------------------------------------
%                        BFGS Method
%--------------------------------------------------------------------

function params = BFGS (x,y,params)
    max_iter = 2500;
    progress = zeros(max_iter,1);
    func = @avg_err;
    grad = @gradients;
    B = eye(length(params));
    
    % initiate g0, d0 and alpha0
    g_minus = grad(x,y,params);
    d = -B*g_minus;
    d = d/norm(d);
    alpha = armijo(func,grad,d,x,y,params);
    params_minus = params;
    
    % find x1
    params = params_minus + alpha*d;
    
    for ii = 1:max_iter
        progress(ii) = func(x,y,params);
        if mod(ii,100)==0
            iteration = ii
            current_error = func(x,y,params)
        end
        % check stopping condition
        g = grad(x,y,params);
        if(abs(g) < eps)
            break
        end
        
        % step
        p = params-params_minus;
        q = g-g_minus;
        s = B*q;
        tau = s.'*q;
        mu = p.'*q;
        v = p/mu-s/tau;
        if (g_minus'*d < g'*d)
            B = B+ p*p.'/mu -s*s.'/tau +tau*(v*v.');
        end
        d = -B*g;
        
        alpha = armijo(func,grad,d,x,y,params); % armijo        
        params_minus = params;
        g_minus = g;
        params = params + alpha*d;

    end
    figure;semilogy(progress);grid on;
    title('BFGS Progress')
end

%--------------------------------------------------------------------
%                           Armijo
%--------------------------------------------------------------------

function alpha = armijo(func, grad, d, x, y, params)
    sigma = 0.25;
    beta = 0.5;
    alpha = 1;
     
    f0 = func(x,y,params);
    df0 = (grad(x,y,params).')*d;

    if(func(x,y,params+alpha*d)-f0<=sigma*alpha*df0)
        while(func(x,y,params+alpha*d)-f0<=sigma*alpha*df0)
            alpha = alpha/beta;
        end
        alpha = alpha*beta;
    else
        while(~(func(x,y,params+alpha*d)-f0<=sigma*alpha*df0))
            if(alpha ==0)
                return
            end
            alpha = alpha*beta;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [b1,b2,b3,W1,W2,W3] = extract_params(params)
    start = 1;
    finish = 4;
    b1 = params(start:finish);
    start = finish+1;
    finish = finish+3;
    b2 = params(start:finish);
    start = finish+1;
    finish = finish+1;
    b3 = params(start:finish);
    start = finish+1;
    finish = finish+8;
    W1 = reshape(params(start:finish),2,4);
    start = finish+1;
    finish = finish+12;
    W2 = reshape(params(start:finish),4,3);
    start = finish+1;
    finish = finish+3;
    W3 = reshape(params(start:finish),3,1);
    
end

function g = gradients(x,y,params)
    epsilon = 1e-8;
    base = eye(length(params));
    g = zeros(length(params),1);
    for ii = 1:length(params)
        g(ii) = (avg_err(x,y,params+ epsilon*base(:,ii))-avg_err(x,y,params- epsilon*base(:,ii)))/(2*epsilon);
    end
end

% tanh
function res = phi(x)
    res = (1 - exp(-2*x)) ./ (1 + exp(-2*x));
end

function res = F(x,params)
    [b1,b2,b3,W1,W2,W3] = extract_params(params);
    res = W3.' * phi(W2.' * phi(W1.' * x + b1) + b2) +b3;
end

function err = avg_err(x,y,params)
    err = mean((F(x,params)-y).^2);
end