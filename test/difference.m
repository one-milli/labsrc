% 2つの画像を読み込む
image1 = cell2mat(struct2cell(load('daruma_F_raw.mat')));
image2 = cell2mat(struct2cell(load('daruma_F_restrict.mat')));

% RGB各チャンネルの差分を計算
diff_red = double(image1(:, :, 1)) - double(image2(:, :, 1));
diff_green = double(image1(:, :, 2)) - double(image2(:, :, 2));
diff_blue = double(image1(:, :, 3)) - double(image2(:, :, 3));

% カラーマップの範囲を計算
caxis_range = max(abs([min(diff_red(:)), max(diff_red(:)), ...
                           min(diff_green(:)), max(diff_green(:)), ...
                           min(diff_blue(:)), max(diff_blue(:))]));

% Red Channelの差分を2次元表示する
figure;
imagesc(diff_red, [-caxis_range, caxis_range]);
colormap('jet');
colorbar;
title('Red Channel Difference');
axis off;
daspect([1 1 1]);

% Green Channelの差分を2次元表示する
figure;
imagesc(diff_green, [-caxis_range, caxis_range]);
colormap('jet');
colorbar;
title('Green Channel Difference');
axis off;
daspect([1 1 1]);

% Blue Channelの差分を2次元表示する
figure;
imagesc(diff_blue, [-caxis_range, caxis_range]);
colormap('jet');
colorbar;
title('Blue Channel Difference');
axis off;
daspect([1 1 1]);
