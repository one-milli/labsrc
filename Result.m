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

                dirname = ['../data/reconst/ADMM_', reconstDate, '/1e', int2str(log10(obj.threshold)), '/'];

                if ~exist(dirname, 'dir')
                    mkdir(dirname);
                end

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, [dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, [dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                end

            else

                dirname = ['../data/reconst/ADMM_', reconstDate, '/'];

                if ~exist(dirname, 'dir')
                    mkdir(dirname);
                end

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, [dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, [dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'], 'BitDepth', 8);
                end

            end

            g_dirname = [dirname, 'graph/'];

            if ~exist(g_dirname, 'dir')
                mkdir(g_dirname);
            end

            x_axis = 1:obj.iters;
            figure(3), plot(x_axis, obj.error);
            exportgraphics(gca, [g_dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '_gr.png']);
        end

    end

end
