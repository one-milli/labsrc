classdef SystemMatrix

    properties
        H_rr
        H_rg
        H_rb
        H_gr
        H_gg
        H_gb
        H_br
        H_bg
        H_bb
    end

    methods

        function obj = SystemMatrix(config)
            hadamardBasis = HadamardBasis(config.getOutputSize);
            U = hadamardBasis.getMatrix();

            disp('Loading capture data...');
            singleColorCaptureR = SingleColorCapture(config, config.getOutputSize, 'R');
            singleColorCaptureG = SingleColorCapture(config, config.getOutputSize, 'G');
            singleColorCaptureB = SingleColorCapture(config, config.getOutputSize, 'B');

            disp('Calculating system matrix...');

            obj.H_rr = (singleColorCaptureR.channelR * U') / (config.getOutputSize ^ 2);
            obj.H_rr(abs(obj.H_rr) <= config.getThreshold) = 0;
            obj.H_rr = sparse(obj.H_rr);

            obj.H_rg = (singleColorCaptureG.channelR * U') / (config.getOutputSize ^ 2);
            obj.H_rg(abs(obj.H_rg) <= config.getThreshold) = 0;
            obj.H_rg = sparse(obj.H_rg);

            obj.H_rb = (singleColorCaptureB.channelR * U') / (config.getOutputSize ^ 2);
            obj.H_rb(abs(obj.H_rb) <= config.getThreshold) = 0;
            obj.H_rb = sparse(obj.H_rb);

            obj.H_gr = (singleColorCaptureR.channelG * U') / (config.getOutputSize ^ 2);
            obj.H_gr(abs(obj.H_gr) <= config.getThreshold) = 0;
            obj.H_gr = sparse(obj.H_gr);

            obj.H_gg = (singleColorCaptureG.channelG * U') / (config.getOutputSize ^ 2);
            obj.H_gg(abs(obj.H_gg) <= config.getThreshold) = 0;
            obj.H_gg = sparse(obj.H_gg);

            obj.H_gb = (singleColorCaptureB.channelG * U') / (config.getOutputSize ^ 2);
            obj.H_gb(abs(obj.H_gb) <= config.getThreshold) = 0;
            obj.H_gb = sparse(obj.H_gb);

            obj.H_br = (singleColorCaptureR.channelB * U') / (config.getOutputSize ^ 2);
            obj.H_br(abs(obj.H_br) <= config.getThreshold) = 0;
            obj.H_br = sparse(obj.H_br);

            obj.H_bg = (singleColorCaptureG.channelB * U') / (config.getOutputSize ^ 2);
            obj.H_bg(abs(obj.H_bg) <= config.getThreshold) = 0;
            obj.H_bg = sparse(obj.H_bg);

            obj.H_bb = (singleColorCaptureB.channelB * U') / (config.getOutputSize ^ 2);
            obj.H_bb(abs(obj.H_bb) <= config.getThreshold) = 0;
            obj.H_bb = sparse(obj.H_bb);

            disp('System matrix calculated.');
        end

%{
         function obj = sparsification(obj, threshold)
            obj.matrix(abs(obj.matrix) <= threshold) = 0;
            obj.matrix = sparse(obj.matrix);
        end
%}

    end

end
