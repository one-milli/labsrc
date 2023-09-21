%% Parameter
object = 'colorchecker';
s = 128; %input
n = 64; %output
eps = 1.0e-4;
% Image range
r1 = 460;
r2 = 920;
c1 = 400;
c2 = 860;

exH = exist('H', 'var');

if exH ~= 1
    % load system matrix
    H = (load('../reconst/H_230601.mat'));
    H = cell2mat(struct2cell(H));
end

% read captured image and stretch
g = imread(['../capture_230516/', object, '.png']);
g = double(imresize(g(r1:r2, c1:c2, :), [s, s])) / 255;
imwrite(g, ['../bef_reconst/cap_230516/', object, '.png'], 'BitDepth', 16)
figure(1), imshow(g);
g_col = reshape(g, [], 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g_col = H' * g_col;
HTH = H' * H;
f_k = double(zeros(n * n * 3, 1));
r_k = g_col - HTH * f_k;
p_k = r_k;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('begin')
count = 1;
tStart = tic;

while (count < 10000)
    alpha_k = dot(p_k, r_k) / dot(p_k, HTH * p_k);
    f_k = f_k + alpha_k * p_k;
    r_k = r_k - alpha_k * HTH * p_k;
    e = norm(r_k);
    disp(['Iteration= ', num2str(count), ',  e = ', num2str(e)]);

    if (e < eps)
        break;
    end

    beta_k = -dot(r_k, HTH * p_k) / dot(p_k, HTH * p_k);
    p_k = r_k + beta_k * p_k;
    count = count + 1;
end

tElapsed = toc(tStart);

% devide RGB
F_r = reshape(f_k(1:n ^ 2, 1), n, n);
F_g = reshape(f_k(n ^ 2 + 1:2 * n ^ 2, 1), n, n);
F_b = reshape(f_k(2 * n ^ 2 + 1:3 * n ^ 2, 1), n, n);

F = double(zeros(n, n, 3));
F(:, :, 1) = F_r;
F(:, :, 2) = F_g;
F(:, :, 3) = F_b;

figure(2), imshow(F);
imwrite(F, ['../reconst/CGM_230516/', object, '.png'], 'BitDepth', 16);

clear count c1 c2 r1 r2 exH eps tStart
