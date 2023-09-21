classdef SystemMatrix

    properties
        matrix
    end

    methods

        function obj = SystemMatrix(config)
            hadamardBasis = HadamardBasis(64);
            U = hadamardBasis.getMatrix();

            disp('Loading capture data...');
            singleColorCaptureR = SingleColorCapture(config, 'R');
            singleColorCaptureG = SingleColorCapture(config, 'G');
            singleColorCaptureB = SingleColorCapture(config, 'B');

            disp('Calculating system matrix...');

            H_rr = (singleColorCaptureR.channelR * U') / (config.getOutputSize ^ 2);
            H_rg = (singleColorCaptureG.channelR * U') / (config.getOutputSize ^ 2);
            H_rb = (singleColorCaptureB.channelR * U') / (config.getOutputSize ^ 2);
            H_gr = (singleColorCaptureR.channelG * U') / (config.getOutputSize ^ 2);
            H_gg = (singleColorCaptureG.channelG * U') / (config.getOutputSize ^ 2);
            H_gb = (singleColorCaptureB.channelG * U') / (config.getOutputSize ^ 2);
            H_br = (singleColorCaptureR.channelB * U') / (config.getOutputSize ^ 2);
            H_bg = (singleColorCaptureG.channelB * U') / (config.getOutputSize ^ 2);
            H_bb = (singleColorCaptureB.channelB * U') / (config.getOutputSize ^ 2);

%{
             H_rg = zeros(16384, 4096);
            H_rb = zeros(16384, 4096);
            H_gr = zeros(16384, 4096);
            H_gb = zeros(16384, 4096);
            H_br = zeros(16384, 4096);
            H_bg = zeros(16384, 4096);
%}

            H_tmp1 = cat(2, H_rr, H_rg);
            H_tmp1 = cat(2, H_tmp1, H_rb);
            H_tmp2 = cat(2, H_gr, H_gg);
            H_tmp2 = cat(2, H_tmp2, H_gb);
            H_tmp3 = cat(2, H_br, H_bg);
            H_tmp3 = cat(2, H_tmp3, H_bb);

            obj.matrix = cat(1, H_tmp1, H_tmp2);
            obj.matrix = cat(1, obj.matrix, H_tmp3);

            disp('System matrix calculated.');
        end

        function obj = sparsification(obj, threshold)
            obj.matrix(abs(obj.matrix) <= threshold) = 0;
            obj.matrix = sparse(obj.matrix);
        end

    end

end
