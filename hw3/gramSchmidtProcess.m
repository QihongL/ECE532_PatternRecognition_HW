%% Gram-Schmidt orthogonalization process
% Input: a matrix V
% Output: a orthogonal matrix Q, where span(col(Q)) == span(col(V))
function Q = gramSchmidtProcess(V)
% read dim and num vectors
dimension = size(V,1);
nCol = size(V,2);
%% if 1st column of A if the zero vector, just eliminate it
while V(:,1) == zeros(dimension,1)
    V = V(:,2:end);     % throw away zero column
    nCol = nCol-1;      % decrement numColumns 
end

%% start Gram-Schmidt process
% normalize the 1st column of A to get the 1st basis vector
Q(:,1) = V(:,1) / norm(V(:,1),2);
% orthogonalize the rest
for j = 2 : nCol
    % compute the projected component
    projection = zeros(dimension,1);
    for i = 1 : j-1
        projection = projection + Q(:,i)' * V(:,j) * Q(:,i);
    end
    % substract to get the orthogonal component
    temp = V(:,j) - projection;
    % if the orthogonal component is not zero vector
    if ~all(temp == 0)
        % normalize the orthogonal component to get the next column of Q
        Q(:,j) = temp / norm(temp);
    end
end
end
