%% Execute ADMM Reconstruction
% Input size, Output size(Hadamard dim), ExpDate, Threshold
config = Config(128, 64, '230516', 1e-2);
objectName = 'capturewhite';
tau = 1e-3;
isSparse = true;
reconstDate = string(datetime("now"), "yyMMdd");
note = "capturewhite";

% read captured image and stretch
g = imread(['../data/capture_', config.getExpDate(), '/', objectName, '.png']);
g = double(imresize(g(460:920, 400:860, :), [config.getInputSize(), config.getInputSize()])) / 255; % 64
% g = double(imresize(g(400:850, 400:850, :), [config.getInputSize(), config.getInputSize()])) / 255; % 128
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

HTH = HTH(systemMatrix, isSparse);

loop1 = [1e1, 1e2, 1e3];
loop2 = [1e-2, 1e-1, 1];

admm = ADMM(config, isSparse);

for i = 1:length(loop1)

    for j = 1:length(loop2)
        mu1 = loop1(i);
        mu2 = loop2(j);
        result = admm.reconstruction(g_col, mu1, mu2, tau, systemMatrix, HTH);
        result.save(objectName, 2, reconstDate, isSparse, note);
    end

end

clear admm;
