%% Generate system matrix
% Input size, Output size, Trim row from, Trim row to, Trim column from, Trim column to, ExpDate
config = Config(128, 64, 460, 920, 400, 860, '230516', 1e-2);

tStart = tic;

% systemMatrix = SystemMatrix(config);
systemMatrix = load(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_origin.mat']).systemMatrix;
systemMatrix = systemMatrix.sparsification(config.getThreshold());

tElapsed = toc(tStart);
disp(['Time elapsed: ' num2str(tElapsed) ' seconds.']);

save(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_1e', int2str(log10(config.getThreshold())), '.mat'], 'systemMatrix', '-v7.3');
