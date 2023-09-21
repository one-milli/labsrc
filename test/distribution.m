matrix1 = image1(:, :, 1);
matrix2 = image2(:, :, 1);

% 0未満の値、0以上1以下の値、1より大きい値のインデックスを取得
less_than_zero1 = matrix1 < 0;
between_zero_and_one1 = matrix1 >= 0 & matrix1 <= 1;
greater_than_one1 = matrix1 > 1;

less_than_zero2 = matrix2 < 0;
between_zero_and_one2 = matrix2 >= 0 & matrix2 <= 1;
greater_than_one2 = matrix2 > 1;

% カラーマップで表示
figure;

subplot(1, 2, 1);
colormap([0 0 1; 0 1 0; 1 0 0]);
imagesc(less_than_zero1 + between_zero_and_one1 * 2 + greater_than_one1 * 3);
colorbar;
xticks(1:64);
yticks(1:64);
xticklabels([]);
yticklabels([]);
title('推定行列(画素値制限なし)R成分 - 行列の成分の値の分布');

subplot(1, 2, 2);
colormap([0 0 1; 0 1 0; 1 0 0]);
imagesc(less_than_zero2 + between_zero_and_one2 * 2 + greater_than_one2 * 3);
colorbar;
xticks(1:64);
yticks(1:64);
xticklabels([]);
yticklabels([]);
title('推定行列(画素値制限あり)R成分 - 行列の成分の値の分布');
