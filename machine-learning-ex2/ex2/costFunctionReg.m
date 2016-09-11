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
c = size(theta,1);


h = sigmoid(X * theta); % 118 x 1

p1 = -y' * log(h);  % 1 x 1
p2 = (1 - y)' * log(1 - h); % 1 x 1

reg = lambda/(2 * m) * sum(theta(2:c,:).^2);

J = 1/m * (p1 - p2) + reg;

grad(1) = 1/m * (X(:,1)' * (h - y) ); % gradient for theta(0)

gp1 = (1/m * (X(:,2:c)' * (h - y)));
gp2 = (theta(2:c,:) * lambda/m);

grad(2:c)= gp1 + gp2 ; % gradient for theta>1


% =============================================================

end
