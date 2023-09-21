classdef SingleColorCapture

    properties (Access = private)
        resizeSize
        outputSize
        rowFrom
        rowTo
        colFrom
        colTo
        expDate
    end

    properties
        baseChannelR
        baseChannelG
        baseChannelB
        channelR
        channelG
        channelB
    end

    methods

        function obj = SingleColorCapture(config, color)
            obj.resizeSize = config.getInputSize;
            obj.outputSize = config.getOutputSize;
            obj.rowFrom = config.getTrimRowFrom;
            obj.rowTo = config.getTrimRowTo;
            obj.colFrom = config.getTrimColFrom;
            obj.colTo = config.getTrimColTo;
            obj.expDate = config.getExpDate;
            obj.channelR = zeros(obj.resizeSize ^ 2, obj.outputSize ^ 2);
            obj.channelG = zeros(obj.resizeSize ^ 2, obj.outputSize ^ 2);
            obj.channelB = zeros(obj.resizeSize ^ 2, obj.outputSize ^ 2);

            imageSrc = ['../data/hadamard_cap_', color, '_', obj.expDate, '/hadamard64_1.png'];
            img = imread(imageSrc);
            img = double(imresize(img(obj.rowFrom:obj.rowTo, obj.colFrom:obj.colTo, :), [obj.resizeSize, obj.resizeSize])) / 255;
            obj.baseChannelR = img(:, :, 1);
            obj.baseChannelG = img(:, :, 2);
            obj.baseChannelB = img(:, :, 3);
            n = (obj.outputSize) ^ 2;

            for k = 1:n

                if mod(k, 500) == 0
                    disp(['Processing ', color, ' ', int2str(k), ' / ', int2str(n)]);
                end

                v0 = imread(['../data/hadamard_cap_', color, '_', obj.expDate, '/hadamard64_', int2str(k), '.png']);
                v0 = double(imresize(v0(obj.rowFrom:obj.rowTo, obj.colFrom:obj.colTo, :), [obj.resizeSize, obj.resizeSize])) / 255;
                v_r = 2 * v0(:, :, 1) - obj.baseChannelR;
                v_g = 2 * v0(:, :, 2) - obj.baseChannelG;
                v_b = 2 * v0(:, :, 3) - obj.baseChannelB;

                obj.channelR(:, k) = double(reshape(v_r, [], 1));
                obj.channelG(:, k) = double(reshape(v_g, [], 1));
                obj.channelB(:, k) = double(reshape(v_b, [], 1));

                if mod(k, n) == 0
                    disp('Done!');
                end

            end

        end

    end

end
