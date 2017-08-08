function x = emailFeatures(word_indices)
%   EMAILFEATURES takes in a word_indices vector and produces a feature vector
%   from the word indices

% Total number of words in the dictionary
n = 1899;

x = zeros(n, 1);

vocabList = getVocabList();
for i=1:n
    if find(word_indices == i)
        x(i) = 1;
    end
end

end
