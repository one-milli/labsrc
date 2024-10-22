m = 128; %control the size of hadamard image
b = 7; % <m-1> can be expressed with <b> bit
% S = ones(m, m, m ^ 2);

n = 1;

for vidx = 0:m - 1

    for uidx = 0:m - 1
        F = getBasisF(uidx, vidx, m, b);
        F = imbinarize(F);
        % S(:, :, n) = F;
        disp(n);

%{
        subplot(m, m, m * vidx + uidx + 1);
        imagesc(F);
        axis image;
        ax = gca;
        ax.XTickLabel = cell(size(ax.XTickLabel)); % Y 軸の目盛り上の数値を削除
        ax.YTickLabel = cell(size(ax.YTickLabel)); % Y 軸の目盛り上の数値を削除
        axis off;
%}

        imwrite(F, ['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(m), '_input/hadamard_', int2str(n), '.png'])
        n = n + 1;
    end

end

%exportgraphics(gcf,'Hadamard64_red/hadamard.pdf','ContentType','image','BackgroundColor','none')
%FileName = strrep(strrep(strcat('Hadamard64_red/hadamard_',datestr(datetime('now')),'.png'),':','_'),' ','_')
%saveas(gcf,FileName)

function F = getBasisF(u, v, m, b)
    F = zeros(m, m);

    for yidx = 0:m - 1

        for xidx = 0:m - 1
            F(yidx + 1, xidx + 1) = (-1) ^ getQ(xidx, yidx, u, v, b);
        end

    end

    F = 1 / m * F;
end

function q = getQ(x, y, u, v, b)
    q = 0;

    for idx = 0:b - 1
        q = q + Bit(idx, x) * getG(idx, u, b) + Bit(idx, y) * getG(idx, v, b);
    end

end

function g = getG(i, u, b)
    g = Bit(b - i, u) + Bit(b - i - 1, u);
end

function bit = Bit(index, u)
    filter = bitshift(1, index);
    flag = bitand(u, filter);

    if flag ~= 0
        bit = 1;
    else
        bit = 0;
    end

end
