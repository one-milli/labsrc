%% Alternating Direction Method of Multipliers
% minf |g-Hf|22 + t|Pf|1 + i(f)

classdef ADMM

    properties
        eps = 0.01;
        config
        D
        DTD
        Psi
        SoftThresh
    end

    methods

        %{
        % Constructor
        %}
        function obj = ADMM(config)
            obj.config = config;
            n = config.getOutputSize();

            D0 = sparse(zeros(n ^ 2));
            Dy = sparse(eye(n ^ 2) - circshift(eye(n ^ 2), [0 1]));
            Dy(n:n:n ^ 2, :) = 0;
            Dx = sparse(eye(n ^ 2) - circshift(eye(n ^ 2), [0 n]));
            Dx(n ^ 2 - n + 1:n ^ 2, :) = 0;

            obj.D = sparse([Dy D0 D0; D0 Dy D0; D0 D0 Dy; Dx D0 D0; D0 Dx D0; D0 D0 Dx]);
            obj.DTD = sparse(obj.D' * obj.D);

            obj.Psi = @(f)(obj.D * f);
            obj.SoftThresh = @(x, t)max(abs(x) - t, 0) .* sign(x);
        end

        %{
        % Reconstruction
        % @return: Result
        %}
        function res = reconstruction(obj, g_col, mu1, mu2, tau, splitH, splitHTH)
            s = obj.config.getInputSize();
            n = obj.config.getOutputSize();
            H = [splitH.H_rr, splitH.H_rg, splitH.H_rb; splitH.H_gr, splitH.H_gg, splitH.H_gb; splitH.H_br, splitH.H_bg, splitH.H_bb];
            HTH = [splitHTH.HTH_rr, splitHTH.HTH_rg, splitHTH.HTH_rb; splitHTH.HTH_gr, splitHTH.HTH_gg, splitHTH.HTH_gb; splitHTH.HTH_br, splitHTH.HTH_bg, splitHTH.HTH_bb];

            R_k = @(W, Z, rho_w, rho_z, G)(H' * (mu1 * G)) + (obj.D' * (mu2 * Z - rho_z)) + mu2 * W - rho_w; %rho_w, rho_z: lagrange multipliers

            % get init matrices
            f = zeros(3 * n ^ 2, 1);
            G = zeros(3 * s ^ 2, 1);
            Z = zeros(6 * n ^ 2, 1);
            W = zeros(3 * n ^ 2, 1);
            rho_z = mu2 * obj.Psi(f);
            rho_w = zeros(3 * n ^ 2, 1);

            temp_r = zeros(n, n);
            temp_g = zeros(n, n);
            temp_b = zeros(n, n);

            disp('Reconstruction begin')
            err = zeros(1, 2);
            iters = 1;
            divmat = 1 / (1 + mu1);
            maxIter = 400;

            disp(['Reconstructing...', ' mu1: ', num2str(mu1), ' mu2: ', num2str(mu2), ' tau: ', num2str(tau)]);
            tStart = tic;
            strictMax = 3;

            while iters <= maxIter
                tStart2 = tic;

                %f_update f<-argmin_f L
                f = (mu1 * HTH + mu2 * obj.DTD + mu2 * speye(3 * n ^ 2)) \ R_k(W, Z, rho_w, rho_z, G);
                %Z_update z<-argmin_z L
                Z = obj.SoftThresh(obj.Psi(f) + rho_z / mu2, tau / mu2); %Proximal operator
                %W_update 0<=W<=1
                W = min(max(f + rho_w / mu2, 0), strictMax);
                %W = f + rho_w / mu2;
                %G_update
                G = divmat * (mu1 * H * f + g_col);
                %eta_update
                rho_z = rho_z + mu2 * (obj.Psi(f) - Z); %Lagrange multipliers associated with Z
                %rho_update
                rho_w = rho_w + mu2 * (f - W); %Lagrange multipliers associated with W

                %calculate error
                [err(1, iters), temp_r, temp_g, temp_b] = obj.calc_err(f, temp_r, temp_g, temp_b);

                tElapsed2 = toc(tStart2);
                disp(['Iteration= ', num2str(iters), ',  e = ', num2str(err(1, iters)), ',  ', num2str(tElapsed2), ' seconds.']);

                if (err(1, iters) < obj.eps || iters == maxIter)
                    break;
                end

                iters = iters + 1;

            end

            tElapsed = toc(tStart);
            disp(['Time elapsed: ' num2str(tElapsed) ' seconds.']);

            resultImage = double(zeros(n, n, 3));
            resultImage(:, :, 1) = (temp_r) ./ strictMax;
            resultImage(:, :, 2) = (temp_g) ./ strictMax;
            resultImage(:, :, 3) = (temp_b) ./ strictMax;

            hf = H * f;

            res = Result(resultImage, err, iters, obj.config.getThreshold(), mu1, mu2, tau, hf);

        end

        %{
            % Calculate error
        %}
        function [error, temp_r, temp_g, temp_b] = calc_err(obj, f, temp_r, temp_g, temp_b)
            n = obj.config.getOutputSize();
            image_r = reshape(f(1:n ^ 2, 1), [n, n]);
            image_g = reshape(f(n ^ 2 + 1:2 * n ^ 2, 1), [n, n]);
            image_b = reshape(f(2 * n ^ 2 + 1:3 * n ^ 2, 1), [n, n]);
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
