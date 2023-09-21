%% Execute ADMM Reconstruction
% Input size, Output size, Trim row from, Trim row to, Trim column from, Trim column to, ExpDate
config = Config(128, 64, 460, 920, 400, 860, '230516', 1e-2);
objectName = 'daruma_edited';
mu1 = 1e2;
mu2 = 1e-2;
tau = 1e-3;

% read captured image and stretch
g = imread(['../data/capture_', config.getExpDate(), '/', objectName, '.png']);
g = double(imresize(g(config.getTrimRowFrom():config.getTrimRowTo(), config.getTrimColFrom():config.getTrimColTo(), :), [config.getInputSize(), config.getInputSize()])) / 255;
imwrite(g, ['../data/bef_reconst/cap_', config.getExpDate(), '/', objectName, '.png'], 'BitDepth', 8)
figure(1), imshow(g);
g_col = reshape(g, [], 1);

%%%%

existH = exist('systemMatrix', 'var');

if existH ~= 1
    % load system matrix
    disp('Loading system matrix...');
    systemMatrix = load(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_1e', int2str(log10(config.getThreshold())), '.mat']).systemMatrix;
    % systemMatrix = load(['../data/systemMatrix/systemMatrix', config.getExpDate(), '_origin.mat']).systemMatrix;
end

disp('Creating ADMM instance...');

admm = ADMM(config, systemMatrix.matrix);

disp(['Reconstructing...', ' mu1: ', num2str(mu1), ' mu2: ', num2str(mu2), ' tau: ', num2str(tau)]);

tStart = tic;
result = admm.reconstruction(g_col, mu1, mu2, tau);
tElapsed = toc(tStart);

disp(['Time elapsed: ' num2str(tElapsed) ' seconds.']);

result.save(objectName, 2);

clear admm;
