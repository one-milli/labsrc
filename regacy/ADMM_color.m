%% Parameter
object = 'daruma';
s = 128; %input
n = 64; %output
eps = 0.01;
% Image range
r1 = 460;
r2 = 920;
c1 = 400;
c2 = 860;
% Hyper-parameters in the ADMM implementation
mu1 = 1.0e2;
mu2 = 1.0e0;
tau = 1.0e-2; %Hyper Parameter

exH = exist('H', 'var');

if exH ~= 1
    % load system matrix
    H = (load('../../data/systemMatrix/H_230608.mat'));
    H = cell2mat(struct2cell(H));
end

HTH = H' * H;

% read captured image and stretch
g = imread(['../../data/capture_230516/', object, '.png']);
g = double(imresize(g(r1:r2, c1:c2, :), [s, s])) / 255;
imwrite(g, ['../../data/bef_reconst/cap_230516/', object, '.png'], 'BitDepth', 16)
figure(1), imshow(g);
g_col = reshape(g, [], 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Dy = eye(n ^ 2) - circshift(eye(n ^ 2), [0 1]);
Dx = eye(n ^ 2) - circshift(eye(n ^ 2), [0 n]);
D0 = zeros(n ^ 2);
D = [Dy D0 D0; D0 Dy D0; D0 D0 Dy; Dx D0 D0; D0 Dx D0; D0 D0 Dx];
DTD = D' * D;
Psi = @(f)(D * f);

SoftThresh = @(x, t)max(abs(x) - t, 0) .* sign(x);
R_k = @(W, Z, rho_w, rho_z, G, xi)H' * (mu1 * G - xi) + D' * (mu2 * Z - rho_z) + mu2 * W - rho_w; %rho_w, rho_z: lagrange multipliers

clear D Dy Dx D0

%get init matrices
G = zeros(3 * s ^ 2, 1);
Z = zeros(6 * n ^ 2, 1);
f = zeros(3 * n ^ 2, 1);
W = zeros(3 * n ^ 2, 1);
xi = zeros(3 * s ^ 2, 1);
rho_z = mu2 * Psi(f);
rho_w = W;

temp_r = zeros(n, n);
temp_g = zeros(n, n);
temp_b = zeros(n, n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('begin')
err = zeros(1, 2);
iters = 0;

divmat = 1 / (1 + mu1);

while (iters < 200)
    iters = iters + 1;
    disp(['Iteration= ', num2str(iters)]);

    %f_update f<-argmin_f L
    tStart = tic;
    f1 = gpuArray(mu1 * HTH + mu2 * DTD + mu2 * eye(3 * n ^ 2));
    f2 = gpuArray(R_k(W, Z, rho_w, rho_z, G, xi));
    f = f1 \ f2;
    tElapsed = toc(tStart);
    disp(tElapsed);
    %Z_update z<-argmin_z L
    Z = SoftThresh(Psi(f) + rho_z / mu2, tau / mu2); %Proximal operator
    %W_update 0<=W<=1
    W = min(max(f + rho_w / mu2, 0), 1);
    %G_update
    G = divmat * (mu1 * H * f + g_col);
    %eta_update
    rho_z = rho_z + mu2 * (Psi(f) - Z); %Lagrange multipliers associated with Z
    %rho_update
    rho_w = rho_w + mu2 * (f - W); %Lagrange multipliers associated with W

    %calculate error
    [err(1, iters), temp_r, temp_g, temp_b] = calc_err(f, temp_r, temp_g, temp_b, n);
    disp(['Iteration= ', num2str(iters), ',  e = ', num2str(err(1, iters))]);

    if (err(1, iters) < eps)
        break;
    end

end

F = double(zeros(n, n, 3));
F(:, :, 1) = temp_r;
F(:, :, 2) = temp_g;
F(:, :, 3) = temp_b;

figure(2), imshow(F, []);
imwrite(F, ['../reconst/ADMM_230802/1e-2', object, '!t', int2str(log10(tau)), ',m1', int2str(log10(mu1)), ',m2', int2str(log10(mu2)), '.png'], 'BitDepth', 16);

x_axis = 1:iters;
figure(3), plot(x_axis, err);
exportgraphics(gca, ['../reconst/ADMM_230802/1e-2', object, '!t', int2str(log10(tau)), ',m1', int2str(log10(mu1)), ',m2', int2str(log10(mu2)), '_gr.png']);

clear iters mu1 mu2 mu2 c1 c2 r1 r2 exH eps object mu1 mu2 n tau tStart

function [error, temp_r, temp_g, temp_b] = calc_err(f, temp_r, temp_g, temp_b, n)
    image_r = reshape(f(1:n ^ 2, 1), [n, n]);
    image_g = reshape(f(n ^ 2 + 1:2 * n ^ 2, 1), [n, n]);
    image_b = reshape(f(2 * n ^ 2 + 1:3 * n ^ 2, 1), [n, n]);
    diff_r = (temp_r - image_r);
    diff_g = (temp_g - image_g);
    diff_b = (temp_b - image_b);
    fenzi_r = norm(diff_r, 'fro');
    fenzi_g = norm(diff_g, 'fro');
    fenzi_b = norm(diff_b, 'fro');
    fenmu_r = norm(image_r, 'fro');
    fenmu_g = norm(image_g, 'fro');
    fenmu_b = norm(image_b, 'fro');

    error = (fenzi_r / fenmu_r) + (fenzi_g / fenmu_g) + (fenzi_b / fenmu_b);
    temp_r = image_r;
    temp_g = image_g;
    temp_b = image_b;
end
