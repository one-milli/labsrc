m = 128; 
b = 7; 

target_n = 8461;

% n から vidx と uidx を逆算
vidx = floor((target_n - 1) / m);
uidx = mod(target_n - 1, m);

F = getBasisF(uidx, vidx, m, b);
F = imbinarize(F);

imwrite(F, ['../data/Hadamard', int2str(m), '_input/hadamard_', int2str(target_n), '.png'])

% 既存の関数定義 (getBasisF, getQ, getG, Bit) はそのまま残す
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
