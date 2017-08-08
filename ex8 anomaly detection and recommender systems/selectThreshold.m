function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%   SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    % Compute the F1 score of choosing epsilon as the threshold and place the
    % value in F1. The code at the end of the loop will compare the F1 score
    % for this choice of epsilon and set it to be the best epsilon if it is
    % better than the current choice of epsilon.

    true_positive  = sum((pval < epsilon) & (yval == 1));
    false_positive = sum((pval < epsilon) & (yval == 0));
    false_negative = sum((pval >= epsilon) & (yval == 1));
    
    if (true_positive + false_positive != 0 && true_positive + false_negative != 0)
        percision = true_positive / (true_positive + false_positive);
        recall    = true_positive / (true_positive + false_negative);
        
        F1 = (2 * percision * recall) / (percision + recall);

        if (F1 > bestF1)
           bestF1 = F1;
           bestEpsilon = epsilon;
        end
    end
end

end
