%% Alternating Direction Method of Multipliers
% minf |g-Hf|22 + t|Pf|1 + i(f)

classdef ADMM

    properties
        eps = 0.01;
        config
        s
        n
        H
        HTH
        D
        DTD
        Psi
        SoftThresh
    end

    methods

        function obj = ADMM(config, H)
            obj.config = config;
            obj.s = config.getInputSize();
            obj.n = config.getOutputSize();
            obj.H = H;

            obj.HTH = H' * H;
            obj.HTH(abs(obj.HTH) <= 1e-4) = 0;
            obj.HTH = sparse(obj.HTH);
            Dy = eye(obj.n ^ 2) - circshift(eye(obj.n ^ 2), [0 1]);
            Dx = eye(obj.n ^ 2) - circshift(eye(obj.n ^ 2), [0 obj.n]);
            D0 = zeros(obj.n ^ 2);
            obj.D = sparse([Dy D0 D0; D0 Dy D0; D0 D0 Dy; Dx D0 D0; D0 Dx D0; D0 D0 Dx]);
            obj.DTD = sparse(obj.D' * obj.D);
            obj.Psi = @(f)(obj.D * f);
            obj.SoftThresh = @(x, t)max(abs(x) - t, 0) .* sign(x);
        end

        function res = reconstruction(obj, g_col, mu1, mu2, tau)
            R_k = @(W, Z, rho_w, rho_z, G, xi)(obj.H' * (mu1 * G - xi)) + (obj.D' * (mu2 * Z - rho_z)) + mu2 * W - rho_w; %rho_w, rho_z: lagrange multipliers

            %get init matrices
            f = zeros(3 * obj.n ^ 2, 1);
            G = zeros(3 * obj.s ^ 2, 1);
            Z = zeros(6 * obj.n ^ 2, 1);
            W = zeros(3 * obj.n ^ 2, 1);
            xi = zeros(3 * obj.s ^ 2, 1);
            rho_z = mu2 * obj.Psi(f);
            rho_w = zeros(3 * obj.n ^ 2, 1);

            temp_r = zeros(obj.n, obj.n);
            temp_g = zeros(obj.n, obj.n);
            temp_b = zeros(obj.n, obj.n);

            disp('Reconstruction begin')
            err = zeros(1, 2);
            iters = 1;
            divmat = 1 / (1 + mu1);
            maxIter = 300;

            while iters <= maxIter
                tStart = tic;

                %f_update f<-argmin_f L
                f = (mu1 * obj.HTH + mu2 * obj.DTD + mu2 * speye(3 * obj.n ^ 2)) \ R_k(W, Z, rho_w, rho_z, G, xi);
                %Z_update z<-argmin_z L
                Z = obj.SoftThresh(obj.Psi(f) + rho_z / mu2, tau / mu2); %Proximal operator
                %W_update 0<=W<=1
                W = min(max(f + rho_w / mu2, 0), 1);
                %G_update
                G = divmat * (mu1 * obj.H * f + g_col);
                %eta_update
                rho_z = rho_z + mu2 * (obj.Psi(f) - Z); %Lagrange multipliers associated with Z
                %rho_update
                rho_w = rho_w + mu2 * (f - W); %Lagrange multipliers associated with W

                %calculate error
                [err(1, iters), temp_r, temp_g, temp_b] = obj.calc_err(f, temp_r, temp_g, temp_b);
                tElapsed = toc(tStart);
                disp(['Iteration= ', num2str(iters), ',  e = ', num2str(err(1, iters)), ',  ', num2str(tElapsed), ' seconds.']);

                if (err(1, iters) < obj.eps || iters == maxIter)
                    break;
                end

                iters = iters + 1;

            end

            resultImage = double(zeros(obj.n, obj.n, 3));
            resultImage(:, :, 1) = temp_r;
            resultImage(:, :, 2) = temp_g;
            resultImage(:, :, 3) = temp_b;

            res = Result(resultImage, err, iters, obj.config.getThreshold(), mu1, mu2, tau);

        end

        function [error, temp_r, temp_g, temp_b] = calc_err(obj, f, temp_r, temp_g, temp_b)
            image_r = reshape(f(1:obj.n ^ 2, 1), [obj.n, obj.n]);
            image_g = reshape(f(obj.n ^ 2 + 1:2 * obj.n ^ 2, 1), [obj.n, obj.n]);
            image_b = reshape(f(2 * obj.n ^ 2 + 1:3 * obj.n ^ 2, 1), [obj.n, obj.n]);
            diff_r = (temp_r - image_r);
            diff_g = (temp_g - image_g);
            diff_b = (temp_b - image_b);
            fenzi_r = norm(diff_r, 'fro');
            fenzi_g = norm(diff_g, 'fro');
            fenzi_b = norm(diff_b, 'fro');
            fenmu_r = norm(image_r, 'fro');
            fenmu_g = norm(image_g, 'fro');
            fenmu_b = norm(image_b, 'fro');

            error = (fenzi_r / fenmu_r) + (fenzi_g / fenmu_g) + (fenzi_b / fenmu_b);
            temp_r = image_r;
            temp_g = image_g;
            temp_b = image_b;
        end

    end

end
