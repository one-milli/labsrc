classdef HTH

    properties
        HTH_rr
        HTH_rg
        HTH_rb
        HTH_gr
        HTH_gg
        HTH_gb
        HTH_br
        HTH_bg
        HTH_bb
    end

    methods

        function obj = HTH(H, isSparse)
            obj.HTH_rr = (H.H_rr)' * H.H_rr + (H.H_gr)' * H.H_gr + (H.H_br)' * H.H_br;
            obj.HTH_rr(abs(obj.HTH_rr) <= 1e-4) = 0;

            if isSparse
                obj.HTH_rr = sparse(obj.HTH_rr);
            end

            obj.HTH_rg = (H.H_rr)' * H.H_rg + (H.H_gr)' * H.H_gg + (H.H_br)' * H.H_bg;
            obj.HTH_rg(abs(obj.HTH_rg) <= 1e-4) = 0;

            if isSparse
                obj.HTH_rg = sparse(obj.HTH_rg);
            end

            obj.HTH_rb = (H.H_rr)' * H.H_rb + (H.H_gr)' * H.H_gb + (H.H_br)' * H.H_bb;
            obj.HTH_rb(abs(obj.HTH_rb) <= 1e-4) = 0;

            if isSparse
                obj.HTH_rb = sparse(obj.HTH_rb);
            end

            obj.HTH_gr = (H.H_rg)' * H.H_rr + (H.H_gg)' * H.H_gr + (H.H_bg)' * H.H_br;
            obj.HTH_gr(abs(obj.HTH_gr) <= 1e-4) = 0;

            if isSparse
                obj.HTH_gr = sparse(obj.HTH_gr);
            end

            obj.HTH_gg = (H.H_rg)' * H.H_rg + (H.H_gg)' * H.H_gg + (H.H_bg)' * H.H_bg;
            obj.HTH_gg(abs(obj.HTH_gg) <= 1e-4) = 0;

            if isSparse
                obj.HTH_gg = sparse(obj.HTH_gg);
            end

            obj.HTH_gb = (H.H_rg)' * H.H_rb + (H.H_gg)' * H.H_gb + (H.H_bg)' * H.H_bb;
            obj.HTH_gb(abs(obj.HTH_gb) <= 1e-4) = 0;

            if isSparse
                obj.HTH_gb = sparse(obj.HTH_gb);
            end

            obj.HTH_br = (H.H_rb)' * H.H_rr + (H.H_gb)' * H.H_gr + (H.H_bb)' * H.H_br;
            obj.HTH_br(abs(obj.HTH_br) <= 1e-4) = 0;

            if isSparse
                obj.HTH_br = sparse(obj.HTH_br);
            end

            obj.HTH_bg = (H.H_rb)' * H.H_rg + (H.H_gb)' * H.H_gg + (H.H_bb)' * H.H_bg;
            obj.HTH_bg(abs(obj.HTH_bg) <= 1e-4) = 0;

            if isSparse
                obj.HTH_bg = sparse(obj.HTH_bg);
            end

            obj.HTH_bb = (H.H_rb)' * H.H_rb + (H.H_gb)' * H.H_gb + (H.H_bb)' * H.H_bb;
            obj.HTH_bb(abs(obj.HTH_bb) <= 1e-4) = 0;

            if isSparse
                obj.HTH_bb = sparse(obj.HTH_bb);
            end

        end

    end

end
