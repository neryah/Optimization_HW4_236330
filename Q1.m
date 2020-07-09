%%%%%%%%%%Q1%%%%%%%%%%%%%
clear all
close all

%% Definitions
epsilon = 1e-8;
b1 = zeros(4,1); b2 = zeros(3,1); b3=0;
W1 = randn(2,4)/sqrt(2); W2 = randn(4,3)/sqrt(4); W3 = randn(3,1)/sqrt(3);
par = struct('phi',@tanH,'phi_t',@tanH_t,'f',@f,'F',@F,'W1',W1,'W2',W2,'W3',W3,'b1',b1,'b2',b2,'b3',b3);

%% Task 1
% create random weights and input
x = 3*randn(2,1);
b1 = 3*randn(4,1); b2 = 3*randn(3,1); b3=3*randn(1);
W1 = randn(2,4)/sqrt(2); W2 = randn(4,3)/sqrt(4); W3 = randn(3,1)/sqrt(3);
par = struct('phi',@tanH,'phi_t',@tanH_t,'f',@f,'F',@F,'W1',W1,'W2',W2,'W3',W3,'b1',b1,'b2',b2,'b3',b3);

% calculate gradients and check error
[W1_g,b1_g,W2_g,b2_g,W3_g,b3_g] = grads(x,par);
Task1(x,par,W1_g,b1_g,W2_g,b2_g,W3_g,b3_g);

%% Task 2
% creating train and test sets
[X1, X2] = meshgrid(-2:.2:2, -2:.2:2);
Y = X1 .* exp(-X1.^2 - X2.^2);
figure; surf(X1, X2, Y)
title('Goal Function')
Ntrain=500;
x_train= 4*rand(2,Ntrain)-2;
Ntest=200;
x_test= 4*rand(2,Ntest)-2;
y_train = par.f(x_train);
y_test = par.f(x_test);

%% Task 3
b1 = zeros(4,1); b2 = zeros(3,1); b3=0;
W1 = randn(2,4)/sqrt(2); W2 = randn(4,3)/sqrt(4); W3 = randn(3,1)/sqrt(3);

params = [reshape(b1,[],1); reshape(b2,[],1); reshape(b3,[],1); reshape(W1,[],1);...
    reshape(W2,[],1); reshape(W3,[],1)];

params = BFGS(x_train,y_train,params);

[b1,b2,b3,W1,W2,W3] = extract_params(params);
par = struct('phi',@tanH,'phi_t',@tanH_t,'f',@f,'F',@F,'W1',W1,'W2',W2,'W3',W3,'b1',b1,'b2',b2,'b3',b3);
network_reconstruction = F(x_test,par).';

l = fit([x_test(1,:)', x_test(2,:)'], network_reconstruction,'linearinterp');
plot( l, [x_test(1,:)', x_test(2,:)'], network_reconstruction);

title('Test Set With Trained Network vs Original Function')
%% Functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               basics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% goal function
function f_x = f(x)
    f_x = x(1,:) .* exp(-x(1,:).^2-x(2,:).^2);
end

% tanh
function res = tanH(x)
    res = (1 - exp(-2*x)) ./ (1 + exp(-2*x));
end

function res = tanH_t(x)
    res = (4.*exp(-2*x)) ./ (1 + exp(-2*x)).^2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           net and grads
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function res = F(x,par)
    res = par.W3.' * par.phi(par.W2.' * par.phi(par.W1.' * x + par.b1) + par.b2) +par.b3;
end

function [W1_g,b1_g,W2_g,b2_g,W3_g,b3_g] = grads(x,par)
    const = 2*(par.F(x,par)-par.f(x));
    q1 = par.phi(par.W1.' * x + par.b1);
    q2 = par.phi(par.W2.' * q1 + par.b2);
    
    b3_g = const;
    W3_g = const*q2;    
    b2_g = const * par.W3.' * diag(par.phi_t(par.W2.'*q1+par.b2));
    W2_g = const*q1 * par.W3.' * diag(par.phi_t(par.W2.'*q1+par.b2));
    b1_g = const * par.W3.' * diag(par.phi_t(par.W2.' * par.phi(par.W1.'*x + par.b1) + par.b2)) * par.W2.' * diag(par.phi_t(par.W1.'*x+par.b1));
    W1_g = const * x * par.W3.' * diag(par.phi_t(par.W2.' * par.phi(par.W1.'*x + par.b1) + par.b2)) * par.W2.' * diag(par.phi_t(par.W1.'*x+par.b1));  
end

function err = psi(x,par)
    err = (par.F(x,par)-par.f(x)).^2;
end

function err = avg_err(x,par)
    err = mean(psi(x,par));
end

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