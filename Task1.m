function [] = Task1(x,par,W1_g,b1_g,W2_g,b2_g,W3_g,b3_g)
    epsilon = 1e-10;
    
    % Test b3
    par_plus = par;
    par_plus.b3 = par.b3+epsilon;
    par_minus = par;
    par_minus.b3 = par.b3-epsilon;
    b3_err = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - b3_g;
    
    % Test W3
    W3_err = zeros(1,numel(par.W3));
    for ii = 1:numel(par.W3)
        par_plus = par;
        par_plus.W3(ii) = par.W3(ii)+epsilon;
        par_minus = par;
        par_minus.W3(ii) = par.W3(ii)-epsilon;
        W3_err(ii) = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - W3_g(ii);
    end
    
    % Test b2
    b2_err = zeros(1,numel(par.b2));
    for ii = 1:numel(par.b2)
        par_plus = par;
        par_plus.b2(ii) = par.b2(ii)+epsilon;
        par_minus = par;
        par_minus.b2(ii) = par.b2(ii)-epsilon;
        b2_err(ii) = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - b2_g(ii);
    end
    
    % Test W2
    W2_err = zeros(1,numel(par.W2));
    for ii = 1:numel(par.W2)
        par_plus = par;
        par_plus.W2(ii) = par.W2(ii)+epsilon;
        par_minus = par;
        par_minus.W2(ii) = par.W2(ii)-epsilon;
        W2_err(ii) = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - W2_g(ii);
    end
    
    % Test b1
    b1_err = zeros(1,numel(par.b1));
    for ii = 1:numel(par.b1)
        par_plus = par;
        par_plus.b1(ii) = par.b1(ii)+epsilon;
        par_minus = par;
        par_minus.b1(ii) = par.b1(ii)-epsilon;
        b1_err(ii) = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - b1_g(ii);
    end
    
    % Test W1
    W1_err = zeros(1,numel(par.W1));
    for ii = 1:numel(par.W1)
        par_plus = par;
        par_plus.W1(ii) = par.W1(ii)+epsilon;
        par_minus = par;
        par_minus.W1(ii) = par.W1(ii)-epsilon;
        W1_err(ii) = (psi(x,par_plus)-psi(x,par_minus))/(2*epsilon) - W1_g(ii);
    end
    
    err = [b3_err, reshape(W3_err,1,[]), reshape(b2_err,1,[]), reshape(W2_err,1,[]),...
        reshape(b1_err,1,[]), reshape(W1_err,1,[])];
    
    figure()
    scatter(1:length(err),abs(err),'*')
    title('Difference between numeric and analitic gradients')
    ylabel('difference[abs]');
    xlabel('parameter index');
    grid on
end

% goal function
function f_x = f(x)
    f_x = x(1) .* exp(-x(1).^2-x(2).^2);
end

function res = F(x,par)
    res = par.W3.' * par.phi(par.W2.' * par.phi(par.W1.' * x + par.b1) + par.b2) +par.b3;
end

function err = psi(x,par)
    err = (par.F(x,par)-par.f(x))^2;
end
