classdef Config

    properties (Access = private)
        inputSize
        outputSize
        expDate
        threshold
    end

    methods

        function obj = Config(inputSize, outputSize, expDate, threshold)
            obj.inputSize = inputSize;
            obj.outputSize = outputSize;
            obj.expDate = expDate;
            obj.threshold = threshold;
        end

        function res = getInputSize(obj)
            res = obj.inputSize;
        end

        function res = getOutputSize(obj)
            res = obj.outputSize;
        end

        function res = getExpDate(obj)
            res = obj.expDate;
        end

        function res = getThreshold(obj)
            res = obj.threshold;
        end

    end

end
