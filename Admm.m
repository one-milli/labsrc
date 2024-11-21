classdef Admm

    properties
        % Parameters
        max_iter = 300;
        tau = 1e0;
        mu1 = 1e-5;
        mu2 = 1e-4;
        mu3 = 1e-4;
        tol = 1e-2;

        % Problem data
        H
        g
        D
        HTH
        DTD
        m
        n

        % ADMM variables
        r
        f
        z
        y
        w
        x
        eta
        rho

        % Error tracking
        err = [];
    end

    methods
        % Constructor
        function obj = Admm(H, g, D)
            obj.H = H;
            obj.g = reshape(g, [], 1);
            obj.D = D;
            obj.HTH = H.' * H;
            obj.DTD = D.' * D;
            [obj.m, obj.n] = size(H);

            obj.r = zeros(obj.n, 1);
            obj.f = ones(obj.n, 1);
            obj.z = zeros(size(D, 1), 1);
            obj.y = zeros(obj.m, 1);
            obj.w = zeros(obj.n, 1);
            obj.x = zeros(obj.m, 1);
            obj.eta = obj.mu2 * (D * obj.f);
            obj.rho = zeros(obj.n, 1);

            if issparse(H)
                fprintf('H is a sparse matrix. Size: %dx%d\n', size(H, 1), size(H, 2));
            end

            if issparse(D)
                fprintf('D is a sparse matrix. Size: %dx%d\n', size(D, 1), size(D, 2));
            end

            if issparse(HTH)
                fprintf('HTH is a sparse matrix. Size: %dx%d\n', size(HTH, 1), size(HTH, 2));
            end

            if issparse(DTD)
                fprintf('DTD is a sparse matrix. Size: %dx%d\n', size(DTD, 1), size(DTD, 2));
            end

            disp('Initialized');
        end

        function s = soft_threshold(~, x, sigma)
            s = max(0, abs(x) - sigma) .* sign(x);
        end

        function obj = compute_obj(obj)
            obj = norm(obj.g - obj.H * obj.f) ^ 2 + obj.tau * norm(obj.D * obj.f, 1);
        end

        function e = compute_err(obj, f_new)
            e = abs(norm(f_new - obj.f) / norm(f_new));
        end

        function obj = update_r(obj)
            obj.r = obj.H.' * (obj.mu1 * obj.y - obj.x) + ...
                obj.D.' * (obj.mu2 * obj.z - obj.eta) + ...
                obj.mu3 * obj.w - obj.rho;
        end

        function obj = update_f(obj)
            A = obj.HTH + obj.mu2 * obj.DTD + obj.mu3 * eye(obj.n);
            obj.f = A \ obj.r;
        end

        function obj = update_z(obj)
            temp = obj.D * obj.f + obj.eta / obj.mu2;
            obj.z = obj.soft_threshold(temp, obj.tau / obj.mu2);
        end

        function obj = update_w(obj)
            temp = obj.f + obj.rho / obj.mu3;
            obj.w = min(max(temp, 0), 1);
        end

        function obj = update_y(obj)
            obj.y = obj.H * obj.f + obj.g;
        end

        function obj = update_x(obj)
            obj.x = obj.x + obj.mu1 * (obj.H * obj.f - obj.y);
        end

        function obj = update_eta(obj)
            obj.eta = obj.eta + obj.mu2 * (obj.D * obj.f - obj.z);
        end

        function obj = update_rho(obj)
            obj.rho = obj.rho + obj.mu3 * (obj.f - obj.w);
        end

        function [f_sol, err] = solve(obj)

            for i = 1:obj.max_iter
                pre_f = obj.f;

                obj = obj.update_r();
                obj = obj.update_f();
                obj = obj.update_z();
                obj = obj.update_w();
                obj = obj.update_y();
                obj = obj.update_x();
                obj = obj.update_eta();
                obj = obj.update_rho();

                error = obj.compute_err(pre_f);
                obj.err = [obj.err; error];

                fprintf('iter = %d, err = %.6f\n', i - 1, error);

                if error < obj.tol
                    break;
                end

            end

            f_sol = obj.f;
            err = obj.err;
        end

    end

end
