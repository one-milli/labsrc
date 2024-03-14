classdef CapturedHadamardBasis

    properties
        baseChannelR
        baseChannelG
        baseChannelB
        channelR
        channelG
        channelB
    end

    methods

        function obj = CapturedHadamardBasis(config, dim, color)
            resizeSize = config.getInputSize;
            outputSize = config.getOutputSize;
            expDate = config.getExpDate;
            obj.channelR = zeros(resizeSize ^ 2, outputSize ^ 2);
            obj.channelG = zeros(resizeSize ^ 2, outputSize ^ 2);
            obj.channelB = zeros(resizeSize ^ 2, outputSize ^ 2);

            img = imread(['../data/hadamard_cap_', color, '_', expDate, '/hadamard', int2str(dim), '_1.png']);
            % img = double(imresize(img, [obj.resizeSize, obj.resizeSize])) / 255; % 128px
            img = double(imresize(img(460:920, 400:860, :), [resizeSize, resizeSize])) / 255; % 64px
            obj.baseChannelR = img(:, :, 1);
            obj.baseChannelG = img(:, :, 2);
            obj.baseChannelB = img(:, :, 3);
            n = (outputSize) ^ 2;

            for k = 1:n

                if mod(k, 512) == 0
                    disp(['Processing ', color, ' ', int2str(k), ' / ', int2str(n)]);
                end

                v0 = imread(['../data/hadamard_cap_', color, '_', expDate, '/hadamard', int2str(dim), '_', int2str(k), '.png']);
                % v0 = double(imresize(v0, [obj.resizeSize, obj.resizeSize])) / 255; % 128px
                v0 = double(imresize(v0(460:920, 400:860, :), [resizeSize, resizeSize])) / 255; % 64px
                v_r = 2 * v0(:, :, 1) - obj.baseChannelR;
                v_g = 2 * v0(:, :, 2) - obj.baseChannelG;
                v_b = 2 * v0(:, :, 3) - obj.baseChannelB;

                obj.channelR(:, k) = double(reshape(v_r, [], 1));
                obj.channelG(:, k) = double(reshape(v_g, [], 1));
                obj.channelB(:, k) = double(reshape(v_b, [], 1));

                if k == n
                    disp('Done!');
                end

            end

        end

    end

end
