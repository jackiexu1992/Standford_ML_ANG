function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possible_value = [0.01 0.03 0.1 0.3 1 3 10 30];
possible_num = size(possible_value,2);

res_error = 99999;

for m = 1:possible_num
   for n = 1:possible_num
       tmp_C = possible_value(m);
       tmp_sigma = possible_value(n);
       model = svmTrain(X, y, tmp_C, @(x1, x2)gaussianKernel(x1, x2, tmp_sigma));
       predictions = svmPredict(model, Xval);
       tmp_error = mean(double(predictions ~= yval));
       if tmp_error < res_error
           C = tmp_C;
           sigma = tmp_sigma;
           res_error = tmp_error;
       end
           
   end
end






% =========================================================================

end
