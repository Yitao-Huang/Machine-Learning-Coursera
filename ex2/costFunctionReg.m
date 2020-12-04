function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

A = sigmoid(X * theta);
B = theta .^2;

for i = 1:m,
  J += - y(i) * log(A(i))-(1 - y(i))* log(1 - A(i));
  for j = 1:size(X, 2)
    grad(j) += (A(i)-y(i)) * X(i, j);
  endfor
end

J = J / m + lambda * (sum(B)-theta(1)^2) / (2 * m);
grad(1) = grad(1) / m;

for k = 2:size(X, 2)
  grad(k) = grad(k) / m + theta(k) * lambda / m;
endfor


% =============================================================

end
