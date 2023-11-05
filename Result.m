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
        hf
    end

    methods

        function obj = Result(image, error, iters, threshold, mu1, mu2, tau, hf)
            obj.image = image;
            obj.image_doubled = imresize(image, 2);
            obj.error = error;
            obj.iters = iters;
            obj.threshold = threshold;
            obj.mu1 = mu1;
            obj.mu2 = mu2;
            obj.tau = tau;
            obj.hf = hf;
        end

        function save(obj, objectName, opt, reconstDate, isSparse, note)

            if isSparse

                dirname = append('../data/reconst/ADMM_', reconstDate, '/1e', int2str(log10(obj.threshold)), '/', note, '/');

                if ~exist(dirname, 'dir')
                    mkdir(dirname);
                end

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, append(dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'), 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, append(dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'), 'BitDepth', 8);
                end

            else

                dirname = append('../data/reconst/ADMM_', reconstDate, '/origin/', note, '/');

                if ~exist(dirname, 'dir')
                    mkdir(dirname);
                end

                if opt == 2
                    figure(2), imshow(obj.image_doubled, []);
                    imwrite(obj.image_doubled, append(dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'), 'BitDepth', 8);
                else
                    figure(2), imshow(obj.image, []);
                    imwrite(obj.image, append(dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '.png'), 'BitDepth', 8);
                end

            end

            g_dirname = append(dirname, 'graph/');

            if ~exist(g_dirname, 'dir')
                mkdir(g_dirname);
            end

            x_axis = 1:obj.iters;
            figure(3), plot(x_axis, obj.error);
            exportgraphics(gca, append(g_dirname, objectName, '!t', int2str(log10(obj.tau)), ',m1', int2str(log10(obj.mu1)), ',m2', int2str(log10(obj.mu2)), '_gr.png'));
        end

    end

end
