%% パラメータの設定
DATA_PATH = '../../OneDrive - m.titech.ac.jp/Lab/data';
% DATA_PATH = '../data';
OBJ_NAME = 'Cameraman';
H_SETTING = 'p-5_lmd-100_m-128';
% H_SETTING = 'gf';
CAP_DATE = '241114';
EXP_DATE = '241118';
n = 128;
m = 128;
PREFIX = '';
% PREFIX = 'int_';

%% Dの作成
D = create_D_mono(n);

%% 画像の読み込みと前処理
image_path = fullfile(DATA_PATH, ['capture_' CAP_DATE], [OBJ_NAME '.png']);
captured = imread(image_path);
% resize
captured = imresize(captured, [m, m]);

if size(captured, 3) == 3
    captured = rgb2gray(captured);
end

captured = cast(captured, 'double') / 255;
g = captured(:);

%% システム行列Hの読み込み
H_path = fullfile(DATA_PATH, EXP_DATE, 'systemMatrix', ['H_matrix_' PREFIX H_SETTING '.mat']);
data = load(H_path);
H = cast(data.H, 'double');
% H が疎行列であることの確認
if issparse(H)
    fprintf('H は疎行列です。サイズ: %dx%d\n', size(H, 1), size(H, 2));
else
    fprintf('H は疎行列ではありません。疎行列に変換します。\n');
    H = sparse(H);
end

% Hの情報を表示
fprintf('H shape: [%d, %d]\n', size(H, 1), size(H, 2));
fprintf('Class of H: %s\n', class(H));
fprintf('Data type of H: %s\n', class(H));

%% 追加の処理（必要に応じて）
disp('システム行列H_sparseの概要:');
whos H

%% ADMMソルバーの実行
admm_solver = Admm(H, g, D);
[f_solution, error_history] = admm_solver.solve();

%% 結果の保存
f_solution = min(max(f_solution, 0), 1);
f_solution = reshape(f_solution, [n, n]);
f_image = cast(f_solution * 255, 'uint8');

tau_log = log10(admm_solver.tau);
mu1_log = log10(admm_solver.mu1);
mu2_log = log10(admm_solver.mu2);
mu3_log = log10(admm_solver.mu3);

reconst_dir = fullfile(DATA_PATH, EXP_DATE, 'reconst');

if ~exist(reconst_dir, 'dir')
    mkdir(reconst_dir);
end

save_filename = sprintf('%s_%s_admm_t-%.1f_m%.1f%.1f%.1f.png', ...
    OBJ_NAME, H_SETTING, tau_log, mu1_log, mu2_log, mu3_log);
save_path = fullfile(reconst_dir, save_filename);
imwrite(f_image, save_path, 'png');
fprintf('Image saved to: %s\n', save_path);

function D = create_D_mono(n)
    % スパースの単位行列を作成
    I = speye(n ^ 2);

    %% Dxの作成
    % 行方向に1シフト（列を右に1シフト）
    shifted_Ix = circshift(I, [0, 1]);

    % 最後の列でのラップを防ぐために特定の行をゼロにする
    rows_to_zero_x = n:n:n ^ 2;
    shifted_Ix(rows_to_zero_x, :) = 0;

    % Dxを計算
    Dx = I - shifted_Ix;

    %% Dyの作成
    % 列方向にnシフト（行を下に1シフト）
    shifted_Iy = circshift(I, [0, n]);

    % 最後のn行でのラップを防ぐためにこれらの行をゼロにする
    shifted_Iy(end - n + 1:end, :) = 0;

    % Dyを計算
    Dy = I - shifted_Iy;

    %% DxとDyを垂直に結合してDを作成
    D = [Dx; Dy];
end
