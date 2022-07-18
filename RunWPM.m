% This function is writen to realize the WPMs solver shown in our paper:
% T. Hong, I. Yavneh, and M. Zibulevsky, ''Solving RED with Weighted Proximal Methods'' ,
% https://arxiv.org/abs/1905.13052.
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   f_denoiser  - call to denoiser, takes two inputs (in, sig_f)
%   input_sigma - noise level
%   params.alpha - initial stepsize and set to be 1 as default.
%   params.lambda - regularization parameter.
%   params.outer_iters - maximal out iterations for WPM.
%   params.effective_sigma - corresponding noise level for calling
%   denoiser.
%   params.isHessianType - the type of updating Hessian matrix. The one
%   used in the paper is 'SR1'. In this file, we also implement other formulations in this function.
%   Note that we do not fully test other formulations for the RED problem, so use it at your own risk for other updating
%   formulations.  
%   params.gamma -  scalling factor for initializing Hessian matrix.
%   params.Max_innerIter -  maximal inner iterations for CG.
%   params.varepsilon

%   orig_im - the original image, used for PSNR evaluation ONLY.
%
% Outputs:
%   im_out - the reconstructed image
%   psnr_out_set - array of PSNR measurements between x_k and orig_im.
%   CPU_time_set - array of run-times taken after the kth iteration.
%   fun_val_set - array of objective values taken after the kth iteration.
%   DenoiserCount - the number of calling denoiser.
%   im_set - array of set of images taken after the kth iteration.
% -------------------------------------------------------------------------
% Note that this function needs additional functions to use.
% See details in the ReadMe file.
% Weighted proximal methods -- Second order methods
% One may add a line search method as a SafeGuard but it may dramatically
% increase the complexity as we claimed in the paper. 
% -------------------------------------------------------------------------
%
function [im_out,psnr_out_set,fun_val_set,CPU_time_set,DenoiserCount,im_set] = ...
    RunWPM(y,ForwardFunc, BackwardFunc,...
    InitEstFunc,input_sigma,params,orig_im)


if nargout>5
    im_set = [];
end
% Get parameters
% some of the parameters contain default value.
alpha = params.alpha; % initial stepsize and set to be 1 first.
lambda = params.lambda;
outer_iters = params.outer_iters;
effective_sigma = params.effective_sigma;
isHessianType = params.isHessianType;
gamma =  params.gamma; % scalling factor for initializing Hessian matrix
Max_innerIter = params.Max_innerIter; % maximal inner iterations for CG
varepsilon = params.varepsilon;

Ht_y = BackwardFunc(y)/(input_sigma^2);

% Initialize parameters
x_est = InitEstFunc(y);
im_size = size(x_est);
im_len = prod(im_size);
psnr_out_set = zeros(outer_iters+1,1);
CPU_time_set = zeros(outer_iters+1,1);
fun_val_set = zeros(outer_iters+1,1);
DenoiserCount = zeros(outer_iters+1,1);
% calculate the initial one
f_x_est = Denoiser(x_est, effective_sigma); % *0.999^(k-1)/1.005^(k-1) -- adaptively change sigma
fun_val = ...
    Cost_Func(y, x_est, ForwardFunc, input_sigma,lambda,f_x_est);
fun_val_set(1) = fun_val;
im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));

psnr_out = ComputePSNR(orig_im, im_out);
psnr_out_set(1) = psnr_out;
CPU_time_set(1) = 0;


DenoiserCount(1) = 0;

% Initialize L and v
t_start = tic;
B = lambda;
Bx = @(x)(B*x);
f_est = Denoiser(x_est, effective_sigma);
g_k = lambda*(x_est-f_est);
x_k = x_est;
x_k_1 = x_est;

for k = 1:outer_iters
    
    if k == 1
        mulAx = @(x)(BackwardFunc(ForwardFunc(x))*(alpha/(input_sigma^2))+Bx(x));
        b = Ht_y + alpha*lambda*f_est+(1-alpha)*lambda*x_est;
    else
        mulAx = @(x)(BackwardFunc(ForwardFunc(x))*(alpha/(input_sigma^2))+reshape(Bx(x(:)),im_size));
        b = alpha*Ht_y + reshape(Bx(x_est(:)),im_size)-alpha*g_k;
    end
    
    x_k_1 = CGAlg(x_est,mulAx,b,Max_innerIter);
    
    if k ~= outer_iters
        
        % do line search
        alpha_temp = alpha;
        while 1
            x_est = x_k+alpha_temp*(x_k_1-x_k);
            f_est = Denoiser(x_est,effective_sigma);
            g_k_1 = lambda*(x_est-f_est);
            fun_val_set(k+1) = 1/(2*input_sigma^2)*norm(ForwardFunc(x_est)-y,'fro')^2+0.5*x_est(:)'*g_k_1(:);
            if fun_val_set(k+1)-fun_val_set(k)<=varepsilon*fun_val_set(k+1)
                DenoiserCount(k+1) = DenoiserCount(k+1)+1;
                x_k_1 = x_est;
                break;
            else
                DenoiserCount(k+1) = DenoiserCount(k+1)+1;
                alpha_temp = 0.5*alpha_temp;
            end
        end
        
        y_k = g_k_1 - g_k;
        s_k = x_est - x_k;
        x_k = x_k_1;
        g_k = g_k_1;
        
        % formulate Hessian. 
        if strcmp(isHessianType,'Rank1Shrinkage')
            tau_BB2 = s_k(:)'*y_k(:)/(norm(y_k(:))^2);
            if tau_BB2<0
                B = lambda;
                Bx = @(x)(B*x);
            else
                H_0 = gamma*tau_BB2;
                temp_1 = s_k(:)-H_0*y_k(:);
                temp_2 = temp_1'*y_k(:);
                if temp_2<=1e-8*norm(y_k(:))*norm(temp_1(:))
                    B = 1/H_0;
                    Bx = @(x)(B*x); 
                else
                    u_k = temp_1/sqrt(temp_2);
                    H_0_inv = (1/H_0)*speye(im_len);
                    H_0_uk = H_0_inv*u_k;
                    B = -(H_0_uk)/(1+u_k'*H_0_uk);
                    Bx = @(x)(H_0_inv*x+B*(H_0_uk'*x));
                end
            end
        elseif strcmp(isHessianType,'Rank1ShrinkageNormal')
            tau_BB2 = s_k(:)'*y_k(:)/(norm(y_k(:))^2);
            tau_BB2 = 1/tau_BB2;
            if tau_BB2<0
                B = lambda;
                Bx = @(x)(B*x);
            else
                B_0 = 1/gamma*tau_BB2;
                temp_1 = y_k(:)-B_0*s_k(:);
                temp_2 = temp_1'*s_k(:);
                if abs(temp_2)<=1e-8*norm(s_k(:))*norm(temp_1(:))
                    B = B_0;
                    Bx = @(x)(B*x);
                else
                    u_k = temp_1;
                    Bx = @(x)(B_0*x+(u_k*(u_k'*x))/temp_2);
                end
            end
        elseif strcmp(isHessianType,'SRI')
            if k==1
                tau_BB2 = s_k(:)'*y_k(:)/(norm(y_k(:))^2);
                tau_BB2 = 1/tau_BB2;
                B_0 =@(x)((1/gamma)*tau_BB2*x);
            else
                B_0 = Bx;
            end
            temp_1 = y_k(:)-B_0(s_k(:));
            temp_2 = temp_1'*s_k(:);
            if abs(temp_2)<=1e-8*norm(s_k(:))*norm(temp_1(:))
                B = B_0;
                Bx = @(x)(B*x);
            else
                if tau_BB2<0
                    B = lambda;
                    Bx = @(x)(B*x);
                else
                    u_k = temp_1;
                    Bx = @(x)(B_0(x)+(u_k*(u_k'*x))/temp_2);
                end
            end
        elseif strcmp(isHessianType,'DFP')
            if k==1
                tau_BB2 = s_k(:)'*y_k(:)/(norm(y_k(:))^2);
                tau_BB2 = 1/tau_BB2;
                B_0 =@(x)((1/gamma)*tau_BB2*x);
            else
                B_0 = Bx;
            end
            rho_k = 1/(y_k(:)'*s_k(:));
            if tau_BB2<0
                B = lambda;
                Bx = @(x)(B*x);
            else
                Bx = @(x)(B_0(x-rho_k*s_k(:)*(y_k(:)'*x))-rho_k*y_k(:)*(s_k(:)'*B_0(x-rho_k*s_k(:)*(y_k(:)'*x)))+rho_k*y_k(:)*(y_k(:)'*x));
            end
            
        elseif strcmp(isHessianType,'BFGS')
            if k==1
                tau_BB2 = s_k(:)'*y_k(:)/(norm(y_k(:))^2);
                tau_BB2 = 1/tau_BB2;
                B_0 =@(x)((1/gamma)*tau_BB2*x);
            else
                B_0 = Bx;
            end
            if tau_BB2<0
                B = lambda;
                Bx = @(x)(B*x);
            else
                Bx = @(x)(B_0(x)-(B_0(s_k(:)*(s_k(:)'*B_0(x))))/(s_k(:)'*B_0(s_k(:)))+(y_k(:)*(y_k(:)'*x))/(y_k(:)'*s_k(:)));
            end
            
        end
    end
    % Find psnr at every iteration
    CPU_time_set(k+1) = toc(t_start);
    psnr_out_set(k+1) = ComputePSNR(orig_im, x_k_1);% psnr(x_est,orig_im,255);
    if nargout>5
        im_set = [im_set x_k_1(:)];
    end
    
    if k == outer_iters
        f_est = Denoiser(x_k_1,effective_sigma);
        fun_val_set(k+1) = Cost_Func(y,x_k_1,ForwardFunc, input_sigma,...
            lambda, f_est);
        DenoiserCount(k+1) = DenoiserCount(k+1)+1;
    end
    
    fprintf('%7i %12.5f %12.5f\n', k, psnr_out_set(k+1),fun_val_set(k+1));
    
    t_start = tic;
end
im_out = x_k_1;
CPU_time_set = cumsum(CPU_time_set);
DenoiserCount = cumsum(DenoiserCount);

return

function x_k_1 = CGAlg(x_k,mulAx,b,MaxIter)

r_k = b-mulAx(x_k);
p_k = r_k;

for iter = 1:MaxIter
    Ap_k = mulAx(p_k);
    alpha_k = r_k(:)'*r_k(:)/(p_k(:)'*Ap_k(:));
    x_k_1 = x_k+alpha_k*p_k;
    if iter<MaxIter
        r_k_1 = r_k-alpha_k*mulAx(p_k);
        beta_k = r_k_1(:)'*r_k_1(:)/(r_k(:)'*r_k(:));
        p_k_1 = r_k_1 + beta_k*p_k;
        
        p_k = p_k_1;
        r_k = r_k_1;
        x_k = x_k_1;
    end 
end

return
