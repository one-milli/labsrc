n = 64; %Hadamard64
s = 128; % Resize
% Image range
r1 = 460;
r2 = 920;
c1 = 400;
c2 = 860;

% Allocate memory
U = zeros(n ^ 2, n ^ 2);
V_rr = zeros(s ^ 2, n ^ 2);
V_rg = zeros(s ^ 2, n ^ 2);
V_rb = zeros(s ^ 2, n ^ 2);
V_gr = zeros(s ^ 2, n ^ 2);
V_gg = zeros(s ^ 2, n ^ 2);
V_gb = zeros(s ^ 2, n ^ 2);
V_br = zeros(s ^ 2, n ^ 2);
V_bg = zeros(s ^ 2, n ^ 2);
V_bb = zeros(s ^ 2, n ^ 2);

capture_white_r = imread('../hadamard_cap_R_230516/hadamard64_1.png');
capture_white_r = double(imresize(capture_white_r(r1:r2, c1:c2, :), [s, s])) / 255;
figure(1), imshow(capture_white_r);
capture_white_rr = capture_white_r(:, :, 1);
capture_white_rg = capture_white_r(:, :, 2);
capture_white_rb = capture_white_r(:, :, 3);

capture_white_g = imread('../hadamard_cap_G_230516/hadamard64_1.png');
capture_white_g = double(imresize(capture_white_g(r1:r2, c1:c2, :), [s, s])) / 255;
figure(2), imshow(capture_white_g);
capture_white_gr = capture_white_g(:, :, 1);
capture_white_gg = capture_white_g(:, :, 2);
capture_white_gb = capture_white_g(:, :, 3);

capture_white_b = imread('../hadamard_cap_B_230516/hadamard64_1.png');
capture_white_b = double(imresize(capture_white_b(r1:r2, c1:c2, :), [s, s])) / 255;
figure(3), imshow(capture_white_b);
capture_white_br = capture_white_b(:, :, 1);
capture_white_bg = capture_white_b(:, :, 2);
capture_white_bb = capture_white_b(:, :, 3);

one = ones(n, n);

for k = 1:4096
    disp(k)
    u0 = (imread(['../Hadamard64_input/hadamard', int2str(k), '.png']));
    u = 2 * u0 - one;
    u = reshape(u, [], 1);
    U(:, k) = u;

    v0_r = imread(['../hadamard_cap_R_230516/hadamard64_', int2str(k), '.png']);
    v0_r = double(imresize(v0_r(r1:r2, c1:c2, :), [s, s])) / 255;
    v0_g = imread(['../hadamard_cap_G_230516/hadamard64_', int2str(k), '.png']);
    v0_g = double(imresize(v0_g(r1:r2, c1:c2, :), [s, s])) / 255;
    v0_b = imread(['../hadamard_cap_B_230516/hadamard64_', int2str(k), '.png']);
    v0_b = double(imresize(v0_b(r1:r2, c1:c2, :), [s, s])) / 255;

    v0_rr = v0_r(:, :, 1);
    v0_rg = v0_r(:, :, 2);
    v0_rb = v0_r(:, :, 3);
    v0_gr = v0_g(:, :, 1);
    v0_gg = v0_g(:, :, 2);
    v0_gb = v0_g(:, :, 3);
    v0_br = v0_b(:, :, 1);
    v0_bg = v0_b(:, :, 2);
    v0_bb = v0_b(:, :, 3);

    % (1, 0) -> (1, -1)
    v_rr = 2 * v0_rr - capture_white_rr;
    v_rg = 2 * v0_rg - capture_white_rg;
    v_rb = 2 * v0_rb - capture_white_rb;
    v_gr = 2 * v0_gr - capture_white_gr;
    v_gg = 2 * v0_gg - capture_white_gg;
    v_gb = 2 * v0_gb - capture_white_gb;
    v_br = 2 * v0_br - capture_white_br;
    v_bg = 2 * v0_bg - capture_white_bg;
    v_bb = 2 * v0_bb - capture_white_bb;

    V_rr(:, k) = double(reshape(v_rr, [], 1));
    V_rg(:, k) = double(reshape(v_rg, [], 1));
    V_rb(:, k) = double(reshape(v_rb, [], 1));
    V_gr(:, k) = double(reshape(v_gr, [], 1));
    V_gg(:, k) = double(reshape(v_gg, [], 1));
    V_gb(:, k) = double(reshape(v_gb, [], 1));
    V_br(:, k) = double(reshape(v_br, [], 1));
    V_bg(:, k) = double(reshape(v_bg, [], 1));
    V_bb(:, k) = double(reshape(v_bb, [], 1));
end

H_rr = (V_rr * U') / (n * n);
H_rg = (V_gr * U') / (n * n);
H_rb = (V_br * U') / (n * n);
H_gr = (V_rg * U') / (n * n);
H_gg = (V_gg * U') / (n * n);
H_gb = (V_bg * U') / (n * n);
H_br = (V_rb * U') / (n * n);
H_bg = (V_gb * U') / (n * n);
H_bb = (V_bb * U') / (n * n);

H_tmp1 = cat(2, H_rr, H_rg);
H_tmp1 = cat(2, H_tmp1, H_rb);
clear H_rr H_rg H_rb;
H_tmp2 = cat(2, H_gr, H_gg);
H_tmp2 = cat(2, H_tmp2, H_gb);
clear H_gr H_gg H_gb;
H_tmp3 = cat(2, H_br, H_bg);
H_tmp3 = cat(2, H_tmp3, H_bb);
clear H_br H_bg H_bb;
H = cat(1, H_tmp1, H_tmp2);
H = cat(1, H, H_tmp3);

%sparsification
H(H <= 1e-4) = 0;
H = sparse(H);

clear H_tmp1 H_tmp2 H_tmp3 one k;
clear capture_white_r capture_white_b capture_white_g capture_white_rr capture_white_rb capture_white_rg capture_white_gr capture_white_gb capture_white_gg capture_white_br capture_white_bb capture_white_bg;
clear u0 u v0_r v0_g v0_b v0_rr v0_rg v0_rb v0_gr v0_gg v0_gb v0_br v0_bg v0_bb v_rr v_rg v_rb v_gr v_gg v_gb v_br v_bg v_bb;
