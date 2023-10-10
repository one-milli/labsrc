%% Execute ADMM Reconstruction
% Input size, Output size(Hadamard dim), ExpDate, Threshold
config = Config(128, 64, '230516', 1e-2);
objectName = 'daruma_edited';
mu1 = 1e2;
mu2 = 1e-2;
tau = 1e-3;
isSparse = true;
reconstDate = '231003';

% read captured image and stretch
g = imread(['../data/capture_', config.getExpDate(), '/', objectName, '.png']);
g = double(imresize(g(400:850, 400:850, :), [config.getInputSize(), config.getInputSize()])) / 255;
% g = double(imresize(g, [config.getInputSize(), config.getInputSize()])) / 255;
% g = double(imresize(g(460:920, 400:860, :), [config.getInputSize(), config.getInputSize()])) / 255;
imwrite(g, ['../data/bef_reconst/cap_', config.getExpDate(), '/', objectName, '.png'], 'BitDepth', 8)
figure(1), imshow(g);
g_col = reshape(g, [], 1);

%%%%

% load system matrix
disp('Loading system matrix...');

if isSparse
    systemMatrix = load(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_1e', int2str(log10(config.getThreshold())), '.mat']).systemMatrix;
else
    systemMatrix = load(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_origin.mat']).systemMatrix;
end

disp('Creating ADMM instance...');

admm = ADMM(config, systemMatrix, isSparse);

disp(['Reconstructing...', ' mu1: ', num2str(mu1), ' mu2: ', num2str(mu2), ' tau: ', num2str(tau)]);

tStart = tic;
result = admm.reconstruction(g_col, mu1, mu2, tau);
tElapsed = toc(tStart);

disp(['Time elapsed: ' num2str(tElapsed) ' seconds.']);

result.save(objectName, 2, reconstDate, isSparse);

clear admm;
