%% Generate sparse system matrix
% Input size, Output size(Hadamard dim), ExpDate, Threshold
config = Config(128, 64, '230516', 1e-2);
isSparse = true;

tStart = tic;

systemMatrix = SystemMatrix(config, isSparse);

tElapsed = toc(tStart);
disp(['Time elapsed: ' num2str(tElapsed) ' seconds.']);

save(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_1e', int2str(log10(config.getThreshold())), '.mat'], 'systemMatrix', '-v7.3');
