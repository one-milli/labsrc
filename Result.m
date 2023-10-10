classdef Result

    properties
        image
        image_doubled
        error
        iters
        threshold
        mu1
        mu2
        tau
    end

    methods

        function obj = Result(image, error, iters, threshold, mu1, mu2, tau)
            obj.image = image;
            obj.image_doubled = imresize(image, 2);
            obj.error = error;
            obj.iters = iters;
            obj.threshold = threshold;
            obj.mu1 = mu1;
            obj.mu2 = mu2;
            obj.tau = tau;
        end

        function save(obj, objectName, opt, reconstDate, isSparse)

            if isSparse

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, ['../data/reconst/ADMM_', reconstDate, '/1e', int2str(log10(obj.threshold)), '/', objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, ['../data/reconst/ADMM_', reconstDate, '/1e', int2str(log10(obj.threshold)), '/', objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                end

            else

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, ['../data/reconst/ADMM_', reconstDate, '/', objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, ['../data/reconst/ADMM_', reconstDate, '/', objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                end

            end

            x_axis = 1:obj.iters;
            figure(3), plot(x_axis, obj.error);
            exportgraphics(gca, ['../data/reconst/ADMM_', reconstDate, '/1e', int2str(log10(obj.threshold)), '/graph/', objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '_gr.png']);
        end

    end

end
