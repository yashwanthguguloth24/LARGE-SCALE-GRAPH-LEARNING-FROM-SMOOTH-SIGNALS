% Refer to GSP ToolBox for more information

%% Actually this feature is the one that allows us to work with big data. 
%% Suppose the graph has n nodes. Classical graph learning [Kalofolias 2016]
%% costs O(n^2) computations. If the mask we give has O(n) edges, then the
%% computation drops to O(n) as well. The question is, how can we compute
%% efficiently a mask of O(n) allowed edges for general data, even if we 
%% don't have prior knowledge? Note that computing the full matrix Z 
%% already costs O(n^2) and we want to avoid it! The solution is Approximate
%% Nearest Neighbors (ANN) that computes an approximate sparse matrix Z with
%% much less computations, roughly O(nlog(n)). This is the idea behind
%% [Kalofolias, Perraudin 2017]

% We compute an approximate Nearest Neighbors graph (using the FLANN
% library through GSP-box)
params_NN.use_flann = 1;
% we ask for an approximate k-NN graph with roughly double number of edges.
% The +1 is because FLANN also counts each node as its own neighbor.
params_NN.k = 2 * k + 1;


clock_flann = tic; 
[indx, indy, dist, ~, ~, ~, NN, Z_sorted] = gsp_nn_distanz(X, X, params_NN);
time_flann = toc(clock_flann);
fprintf('Time for FLANN: %.3f seconds\n', toc(clock_flann));

% gsp_nn_distanz gives distance matrix in a form ready to use with sparse:
Z_sp = sparse(indx, indy, dist.^2, n, n, params_NN.k * n * 2);
% symmetrize the distance matrix
Z_sp = gsp_symmetrize(Z_sp, 'full');
% get rid or first row that is zero (each node has 0 distance from itself)
Z_sorted = Z_sorted(:, 2:end).^2;   % first row is zero

% Note that FLANN returns Z already sorted, that further saves computation
% for computing the parameter theta.
Z_is_sorted = true;
theta = gsp_compute_graph_learning_theta(Z_sorted, k, 0, Z_is_sorted);

params.edge_mask = Z_sp > 0;
params.fix_zeros = 1;

[W3, info_3] = gsp_learn_graph_log_degrees(Z_sp * theta, 1, 1, params);
W3(W3<1e-4) = 0;

fprintf('Relative difference between two solutions: %.4f\n', norm(W-W2, 'fro')/norm(W, 'fro'));
% Note that the learned graph is sparser than the mask we gave as input.

figure; subplot(1, 2, 1); spy(W3); title('W_3'); subplot(1, 2, 2), spy(params.edge_mask); title('edge mask');
