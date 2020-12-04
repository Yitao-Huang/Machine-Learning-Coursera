function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

A = X * theta - y;

% Compute Cost 
A = A.^2;
theta_reg = theta.^2;
J += sum(A) / (m * 2) + lambda * sum(theta_reg(2:end)) / (2 * m);

% Compute Gradient
B = zeros(size(theta,1),1);
A = X * theta - y;

for i = 1:size(theta,1),
  B(i) += sum(A' * X(:, i));
endfor
    
grad(1,1) =  B(1) / m;  

for i = 2:size(theta,1),
  grad(i,1) = B(i) / m + lambda * theta(i,1) / m;
endfor

% =========================================================================

grad = grad(:);

end
