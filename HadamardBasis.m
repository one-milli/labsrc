classdef HadamardBasis

    properties (Access = private)
        matrix
    end

    methods

        %{
            constructor
            @param {int} dim
            @return {object} HadamardBasis
        %}
        function obj = HadamardBasis(dim)
            one = ones(dim, dim);

            for k = 1:dim ^ 2
                u0 = imread(['../data/Hadamard', int2str(dim), '_input/hadamard', int2str(k), '.png']);
                u = reshape(2 * u0 - one, [], 1);
                obj.matrix(:, k) = u;
            end

        end

        %{
            getter
        %}
        function res = getMatrix(obj)
            res = obj.matrix;
        end

    end

end
