% 必要な場合は、readNPY関数へのパスを追加してください
% addpath('path_to_readNPY_function');

%% パラメータの設定
DATA_PATH = '../data';
OBJ_NAME = 'Cameraman';
H_SETTING = 'gf';
CAP_DATE = '241114';
EXP_DATE = '241118';
n = 128;
m = 255;
PREFIX = 'int_';

%% Dの作成
D = create_D_mono(n);

%% 画像の読み込みと前処理
image_path = fullfile(DATA_PATH, ['capture_' CAP_DATE], [OBJ_NAME '.png']);
captured = imread(image_path);

if size(captured, 3) == 3
    captured = rgb2gray(captured);
end

captured = cast(captured, 'single');
g = captured(:);

%% システム行列Hの読み込み
H_path = fullfile(DATA_PATH, EXP_DATE, 'systemMatrix', ['H_matrix_' PREFIX H_SETTING '.npy']);

% .npyファイルの読み込み
% readNPY関数が必要です。ダウンロードしてMATLABパスに追加してください。
H = sparse(readNPY(H_path));

% Hの情報を表示
fprintf('H shape: [%d, %d]\n', size(H, 1), size(H, 2));
fprintf('Class of H: %s\n', class(H));
fprintf('Data type of H: %s\n', class(H));

%% 追加の処理（必要に応じて）
disp('システム行列H_sparseの概要:');
whos H

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
