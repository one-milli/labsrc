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

        function obj = ADMM(config, H, isSparse)
            obj.config = config;
            obj.s = config.getInputSize();
            obj.n = config.getOutputSize();

            H_tmp1 = sparse(cat(2, H.H_rr, H.H_rg));
            H_tmp1 = sparse(cat(2, H_tmp1, H.H_rb));
            H_tmp2 = sparse(cat(2, H.H_gr, H.H_gg));
            H_tmp2 = sparse(cat(2, H_tmp2, H.H_gb));
            H_tmp3 = sparse(cat(2, H.H_br, H.H_bg));
            H_tmp3 = sparse(cat(2, H_tmp3, H.H_bb));

            obj.H = sparse(cat(1, H_tmp1, H_tmp2));
            clear H_tmp1 H_tmp2;
            obj.H = sparse(cat(1, obj.H, H_tmp3));
            clear H_tmp3;

            obj.H = sparse(obj.H);

            if isSparse
                HTH_rr = (H.H_rr)' * H.H_rr + (H.H_gr)' * H.H_gr + (H.H_br)' * H.H_br;
                HTH_rr(abs(HTH_rr) <= 1e-4) = 0;
                HTH_rr = sparse(HTH_rr);
                HTH_rg = (H.H_rr)' * H.H_rg + (H.H_gr)' * H.H_gg + (H.H_br)' * H.H_bg;
                HTH_rg(abs(HTH_rg) <= 1e-4) = 0;
                HTH_rg = sparse(HTH_rg);
                HTH_rb = (H.H_rr)' * H.H_rb + (H.H_gr)' * H.H_gb + (H.H_br)' * H.H_bb;
                HTH_rb(abs(HTH_rb) <= 1e-4) = 0;
                HTH_rb = sparse(HTH_rb);
                HTH_gr = (H.H_rg)' * H.H_rr + (H.H_gg)' * H.H_gr + (H.H_bg)' * H.H_br;
                HTH_gr(abs(HTH_gr) <= 1e-4) = 0;
                HTH_gr = sparse(HTH_gr);
                HTH_gg = (H.H_rg)' * H.H_rg + (H.H_gg)' * H.H_gg + (H.H_bg)' * H.H_bg;
                HTH_gg(abs(HTH_gg) <= 1e-4) = 0;
                HTH_gg = sparse(HTH_gg);
                HTH_gb = (H.H_rg)' * H.H_rb + (H.H_gg)' * H.H_gb + (H.H_bg)' * H.H_bb;
                HTH_gb(abs(HTH_gb) <= 1e-4) = 0;
                HTH_gb = sparse(HTH_gb);
                HTH_br = (H.H_rb)' * H.H_rr + (H.H_gb)' * H.H_gr + (H.H_bb)' * H.H_br;
                HTH_br(abs(HTH_br) <= 1e-4) = 0;
                HTH_br = sparse(HTH_br);
                HTH_bg = (H.H_rb)' * H.H_rg + (H.H_gb)' * H.H_gg + (H.H_bb)' * H.H_bg;
                HTH_bg(abs(HTH_bg) <= 1e-4) = 0;
                HTH_bg = sparse(HTH_bg);
                HTH_bb = (H.H_rb)' * H.H_rb + (H.H_gb)' * H.H_gb + (H.H_bb)' * H.H_bb;
                HTH_bb(abs(HTH_bb) <= 1e-4) = 0;
                HTH_bb = sparse(HTH_bb);

                HTH_tmp1 = cat(2, HTH_rr, HTH_rg);
                HTH_tmp1 = cat(2, HTH_tmp1, HTH_rb);
                clear HTH_rr HTH_rg HTH_rb;
                HTH_tmp2 = cat(2, HTH_gr, HTH_gg);
                HTH_tmp2 = cat(2, HTH_tmp2, HTH_gb);
                clear HTH_gr HTH_gg HTH_gb;
                HTH_tmp3 = cat(2, HTH_br, HTH_bg);
                HTH_tmp3 = cat(2, HTH_tmp3, HTH_bb);
                clear HTH_br HTH_bg HTH_bb;
                obj.HTH = cat(1, HTH_tmp1, HTH_tmp2);
                obj.HTH = cat(1, obj.HTH, HTH_tmp3);
                clear HTH_tmp1 HTH_tmp2 HTH_tmp3;

                obj.HTH = sparse(obj.HTH);
%{
                 obj.HTH = H' * H;
                obj.HTH(abs(obj.HTH) <= 1e-4) = 0;
                obj.HTH = sparse(obj.HTH);
%}

            else
                obj.HTH = H' * H;
            end

            Dy = sparse(eye(obj.n ^ 2) - circshift(eye(obj.n ^ 2), [0 1]));
            Dx = sparse(eye(obj.n ^ 2) - circshift(eye(obj.n ^ 2), [0 obj.n]));
            D0 = sparse(zeros(obj.n ^ 2));
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
