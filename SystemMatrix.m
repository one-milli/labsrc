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

        function obj = SystemMatrix(config, isSparse)
            outputSize = config.getOutputSize;
            hadamardBasis = HadamardBasis(config.getOutputSize);
            U = hadamardBasis.getMatrix();

            disp('Loading capture data...');
            CapturedHadamardBasisR = CapturedHadamardBasis(config, outputSize, 'R');
            CapturedHadamardBasisG = CapturedHadamardBasis(config, outputSize, 'G');
            CapturedHadamardBasisB = CapturedHadamardBasis(config, outputSize, 'B');

            disp('Calculating system matrix...');
            obj.H_rr = (CapturedHadamardBasisR.channelR * U') / (outputSize ^ 2);
            obj.H_rr(abs(obj.H_rr) <= config.getThreshold) = 0;

            if isSparse
                obj.H_rr = sparse(obj.H_rr);
            end

            obj.H_rg = (CapturedHadamardBasisG.channelR * U') / (outputSize ^ 2);
            obj.H_rg(abs(obj.H_rg) <= config.getThreshold) = 0;

            if isSparse
                obj.H_rg = sparse(obj.H_rg);
            end

            obj.H_rb = (CapturedHadamardBasisB.channelR * U') / (outputSize ^ 2);
            obj.H_rb(abs(obj.H_rb) <= config.getThreshold) = 0;

            if isSparse
                obj.H_rb = sparse(obj.H_rb);
            end

            obj.H_gr = (CapturedHadamardBasisR.channelG * U') / (outputSize ^ 2);
            obj.H_gr(abs(obj.H_gr) <= config.getThreshold) = 0;

            if isSparse
                obj.H_gr = sparse(obj.H_gr);
            end

            obj.H_gg = (CapturedHadamardBasisG.channelG * U') / (outputSize ^ 2);
            obj.H_gg(abs(obj.H_gg) <= config.getThreshold) = 0;

            if isSparse
                obj.H_gg = sparse(obj.H_gg);
            end

            obj.H_gb = (CapturedHadamardBasisB.channelG * U') / (outputSize ^ 2);
            obj.H_gb(abs(obj.H_gb) <= config.getThreshold) = 0;

            if isSparse
                obj.H_gb = sparse(obj.H_gb);
            end

            obj.H_br = (CapturedHadamardBasisR.channelB * U') / (outputSize ^ 2);
            obj.H_br(abs(obj.H_br) <= config.getThreshold) = 0;

            if isSparse
                obj.H_br = sparse(obj.H_br);
            end

            obj.H_bg = (CapturedHadamardBasisG.channelB * U') / (outputSize ^ 2);
            obj.H_bg(abs(obj.H_bg) <= config.getThreshold) = 0;

            if isSparse
                obj.H_bg = sparse(obj.H_bg);
            end

            obj.H_bb = (CapturedHadamardBasisB.channelB * U') / (outputSize ^ 2);
            obj.H_bb(abs(obj.H_bb) <= config.getThreshold) = 0;

            if isSparse
                obj.H_bb = sparse(obj.H_bb);
            end

            disp('System matrix calculated.');
        end

    end

end
