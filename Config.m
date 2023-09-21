classdef Config

    properties (Access = private)
        inputSize
        outputSize
        trimRowFrom
        trimRowTo
        trimColFrom
        trimColTo
        expDate
        threshold
    end

    methods

        function obj = Config(inputSize, outputSize, trimRowFrom, trimRowTo, trimColFrom, trimColTo, expDate, threshold)
            obj.inputSize = inputSize;
            obj.outputSize = outputSize;
            obj.trimRowFrom = trimRowFrom;
            obj.trimRowTo = trimRowTo;
            obj.trimColFrom = trimColFrom;
            obj.trimColTo = trimColTo;
            obj.expDate = expDate;
            obj.threshold = threshold;
        end

        function res = getInputSize(obj)
            res = obj.inputSize;
        end

        function res = getOutputSize(obj)
            res = obj.outputSize;
        end

        function res = getTrimRowFrom(obj)
            res = obj.trimRowFrom;
        end

        function res = getTrimRowTo(obj)
            res = obj.trimRowTo;
        end

        function res = getTrimColFrom(obj)
            res = obj.trimColFrom;
        end

        function res = getTrimColTo(obj)
            res = obj.trimColTo;
        end

        function res = getExpDate(obj)
            res = obj.expDate;
        end

        function res = getThreshold(obj)
            res = obj.threshold;
        end

    end

end
