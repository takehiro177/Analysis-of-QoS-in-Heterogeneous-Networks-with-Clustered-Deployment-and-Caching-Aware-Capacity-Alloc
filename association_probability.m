
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% HetNet Parameters %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% configulation window
sq_m = 2000*2000;

% power
P_1 = 3;
P_2 = 13;
P_3 = 193;

% Path loss
alpha = 4;

% cache eneabled probability
%a = 0.1;
a = 0.05:0.025:0.30;

expand = zeros(1,11)+1;

% intensity
lambda_0 = 1000/(pi*1000^2).*expand; % remov pi
lambda_1 = lambda_0.*a;
lambda_2 = 3/(pi*1000^2).*expand; % intensity of parental pp
m_bar = 10*expand;
lambda_3 = 3/(pi*1000^2).*expand;
lambda_2_ppp = lambda_2.*m_bar;  % intensity of sbs when ppp


% TPP Parameters
sigma = 250;

%%%%%%%%%%%%%%%%%%%%
% Association Prob %
%%%%%%%%%%%%%%%%%%%%
% PPP-PPP
% Association to 1
P_i = P_1;
lambda_i = lambda_1;
p_tasso1 = ((lambda_1./lambda_i)*(P_1/P_i).^(2/alpha)+(lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Tier Association Probability [to 1]: %.11f \n', p_tasso1)
% Association to 2
P_i = P_2;
lambda_i = lambda_2_ppp;
p_tasso2 = ((lambda_1./lambda_i)*(P_1/P_i).^(2/alpha)+(lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Tier Association Probability [to 2]: %.11f \n', p_tasso2)
% Association to 3
P_i = P_3;
lambda_i = lambda_3;
p_tasso3 = ((lambda_1./lambda_i)*(P_1/P_i).^(2/alpha)+(lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Tier Association Probability [to 3]: %.11f \n', p_tasso3)

%{
% PPP-PPCP
% Association to 1
% param
P_i = P_1;
lambda_i = lambda_1;
% func
pgfl_Q1 = @(r,z) -lambda_2.*2*pi*z.*(1-exp(-m_bar.*(1-marcumq(z/sigma,(P_2/P_i).^(1/alpha)*r/sigma,1))));
% calc
tasso1_f = @(r) exp(integral(@(z) pgfl_Q1(r,z), 0, Inf,'ArrayValued',true))*2*pi.*lambda_i*r.*exp(-pi.*(lambda_1.*(P_1/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*r^2);
tasso1 = integral(@(r) tasso1_f(r), 0, Inf,'ArrayValued',true, 'RelTol', 1e-3, 'AbsTol', 1e-3);

% Association to 2
% func
A_sumprod1 = @(r,z) m_bar.*lambda_2.*2*pi*z*1/sigma*ricepdf_(r,z,sigma).*exp(-m_bar.*(1-marcumq(z/sigma,(P_2/P_2).^(1/alpha)*r/sigma,1)));
A_sumprod2_in = @(r,z) -lambda_2.*2*pi*z.*(1-exp(-m_bar.*(1-marcumq(z/sigma,(P_2/P_2).^(1/alpha)*r/sigma,1)))); 
% calc
tasso2_f = @(r) integral(@(z) A_sumprod1(r,z) , 0, Inf,'ArrayValued',true) .* exp( integral(@(z) A_sumprod2_in(r,z), 0, Inf,'ArrayValued',true)) .* exp(-pi.*(lambda_1.*(P_1/P_2).^(2/alpha)+lambda_3.*(P_3/P_2).^(2/alpha))*r^2);
tasso2 = integral(@(r) tasso2_f(r), 0, Inf,'ArrayValued',true, 'RelTol', 1e-3, 'AbsTol', 1e-3);

% Association to 3
% param
P_i = P_3;
lambda_i = lambda_3;
% func
pgfl_Q1 = @(r,z) -lambda_2.*2*pi*z.*(1-exp(-m_bar.*(1-marcumq(z/sigma,(P_2/P_i).^(1/alpha)*r/sigma,1))));  
% calc
tasso3_f = @(r) exp(integral(@(z) pgfl_Q1(r,z), 0, Inf,'ArrayValued',true)).*2*pi.*lambda_i*r.*exp(-pi.*(lambda_1.*(P_1/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*r^2);
tasso3 = integral(@(r) tasso3_f(r), 0, Inf,'ArrayValued',true, 'RelTol', 1e-3, 'AbsTol', 1e-3);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Ordered tier association probability  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PPP-PPP
% (Order of 2)
% 2>3
P_i = P_2;
lambda_i = lambda_2_ppp;
p_oasso23 = ((lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Ordered Two Tier Association Probability [3 < 2]: %.11f \n', p_oasso23)
% 3>2
P_i = P_3;
lambda_i = lambda_3;
p_oasso32 = ((lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Ordered Two Tier Association Probability [2 < 3]: %.11f \n', p_oasso32)

% (Order of 3)
% 1>2>3
P_k = P_3;
lambda_k = lambda_3;
P_j = P_2;
lambda_j = lambda_2_ppp;
P_i = P_1;
lambda_i = lambda_1;
p_3oasso123 = (1+(lambda_k./lambda_j)*(P_k/P_j).^(2/alpha)).^(-1).*((lambda_1./lambda_i)*(P_1/P_i).^(2/alpha)+(lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Ordered Three Tier Association Probability [3 < 2 < 1]: %.11f \n', p_3oasso123)

%1>3>2
P_k = P_2;
lambda_k = lambda_2_ppp;
P_j = P_3;
lambda_j = lambda_3;
P_i = P_1;
lambda_i = lambda_1;
p_3oasso132 = (1+(lambda_k./lambda_j)*(P_k/P_j).^(2/alpha)).^(-1).*((lambda_1./lambda_i)*(P_1/P_i).^(2/alpha)+(lambda_2_ppp./lambda_i)*(P_2/P_i).^(2/alpha)+(lambda_3./lambda_i)*(P_3/P_i).^(2/alpha)).^(-1);
fprintf('(PPP) Ordered Three Tier Association Probability [2 < 3 < 1]: %.11f \n', p_3oasso132)


% PPP-PPCP
% (Order of 2)
%### Library 
% sum-product for ordered tier association probability
O_sumprod1 = @(r,z) m_bar*lambda_2*2*pi*z*1/sigma*ricepdf_(r,z,sigma)*exp(-m_bar*(1-marcumq(z/sigma,r/sigma,1)));
O_sumprod2_in = @(r,z) -lambda_2*2*pi*z*(1-exp(-m_bar*(1-marcumq(z/sigma,r/sigma,1))));  
%O_SUMPROD = @(r) m_bar*integral(@(z) O_sumprod1(r,z) , 0, Inf,'ArrayValued',true) * exp( integral(@(z) O_sumprod2_in(r,z), 0, Inf,'ArrayValued',true));
%result = @(r_t) integral(@(r) SUMPROD(r), r_t , Inf,'ArrayValued',true);

% 2>3
oasso23_f = @(r_j) 2*pi*lambda_3*r_j*exp(-pi*lambda_3*r_j^2);
oasso23_ff = @(r_i) integral(@(z) O_sumprod1(r_i,z) , 0, Inf,'ArrayValued',true) * exp( integral(@(z) O_sumprod2_in(r_i,z), 0, Inf,'ArrayValued',true))*integral(@(r) oasso23_f(r), (P_3/P_2)^(1/alpha)*r_i, Inf,'ArrayValued',true);
oasso23 = integral(@(r_i) oasso23_ff(r_i), 0, Inf,'ArrayValued',true, 'RelTol', 1e-3, 'AbsTol', 1e-3);
fprintf('(PPCP) Orderd Two Tier Association Probability [3 < 2]: %.11f \n', oasso23)
% 3>2
oasso32_f = @(r_j) integral(@(z) O_sumprod1(r_j,z) , 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2) * exp( integral(@(z) O_sumprod2_in(r_j,z), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2));
oasso32_ff = @(r_i) integral(@(r_j) oasso32_f(r_j), (P_2/P_3)^(1/alpha)*r_i, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2) * 2*pi*lambda_3*r_i*exp(-pi*lambda_3*r_i^2);
oasso32 = integral(@(r_i) oasso32_ff(r_i), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2);
fprintf('(PPCP) Orderd Two Tier Association Probability [2 < 3]: %.11f \n', oasso32)


% (Order of 3)
% 1>2>3
oasso123_f = @(r_k) 2*pi*lambda_3*r_k*exp(-pi*lambda_3*r_k^2);
oasso123_ff = @(r_j) integral(@(r_k) oasso123_f(r_k), (P_3/P_2)^(1/alpha)*r_j, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2)*integral(@(z) O_sumprod1(r_j,z) , 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2) * exp( integral(@(z) O_sumprod2_in(r_j,z), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2));
oasso123_fff = @(r_i) integral(@(r_j) oasso123_ff(r_j), (P_2/P_1)^(1/alpha)*r_i, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2)*2*pi*lambda_1*r_i*exp(-pi*lambda_1*r_i^2);
oasso123 = integral(@(r_i) oasso123_fff(r_i), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2);
fprintf('(PPCP) Orderd Three Tier Association Probability [3 < 2 < 1]: %.11f \n', oasso123)
% 1>3>2
%{
oasso132_f = @(r_k) integral(@(z) O_sumprod1(r_k,z) , 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2)* exp(integral(@(z) O_sumprod2_in(r_k,z), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2));
oasso132_ff = @(r_j) integral(@(r_k) oasso132_f(r_k), (P_2/P_3)^(1/alpha)*r_j, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2)*2*pi*lambda_3*r_j*exp(-pi*lambda_3*r_j^2);
oasso132_fff = @(r_i) integral(@(r_j) oasso132_ff(r_j), (P_3/P_1)^(1/alpha)*r_i, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2)*2*pi*lambda_1*r_i*exp(-pi*lambda_1*r_i^2);
oasso132 = integral(@(r_i) oasso132_fff(r_i), 0, Inf,'ArrayValued',true, 'RelTol', 1e-2, 'AbsTol', 1e-2);
%}
oasso132 = tasso1 - oasso123;
fprintf('(PPCP) Orderd Three Tier Association Probability [2 < 3 < 1]: %.11f \n', oasso132)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Average Ergodic Rate PPP-PPP/PPCP %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%
% Zipf %
%%%%%%%%
gamma = 0.8;
M_1 = 10;
M_2 = 50;
N = 200;
syms k
zipf_deno = @(g) vpa(symsum(1/(k^g),k,1,N));
zipf = @(g) (1/k^g) / zipf_deno(g);
zipf_cdf_1TOm1 = @(g) symsum(zipf(g),k,1,M_1);
zipf_cdf_1TOm2 = @(g) symsum(zipf(g),k,1,M_2);
zipf_cdf_m1TOm2 = @(g) symsum(zipf(g),k,M_1 + 1,M_2);
zipf_cdf_m1TON = @(g) symsum(zipf(g),k,M_1 + 1,N);
zipf_cdf_m2TON = @(g) symsum(zipf(g),k,M_2 + 1,N);
F_1_M1 =arrayfun(@(g) zipf_cdf_1TOm1(g),gamma);


%---------------------------------------------------------------------------------------------------
% PPP-PPP
active_lambda_1 = (1-a).*lambda_0.*p_tasso1*F_1_M1;
lambda_1_d = double(min(lambda_0.*a, active_lambda_1));
             

%%%%%%%%%%
% Case 1 %
%%%%%%%%%%

%---------------------------------------------------------------------------------------------------
% to 1
P_i = P_1;
M_y1ppp = @(x,s) exp(-2*pi.*lambda_2_ppp*(P_2/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_2/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_1_d*(P_1/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_1/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_3*(P_3/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_3/P_i)^(1/4-1)*x)));
M_xy1ppp = @(x,s) 1./(1+s*x.^(-alpha)).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) (M_y1ppp(x,s) - M_xy1ppp(x,s))/s, 0, Inf,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_1_1_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_1 * x./p_tasso1 .* exp(-pi.*(lambda_1.*(P_1/P_i).^(2/alpha)+lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2);
U_1_1_ppp = integral(@(x) U_1_1_f_ppp(x), 0, Inf, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
%---------------------------------------------------------------------------------------------------
% to 2

P_i = P_2;
M_y1ppp = @(x,s) exp(-2*pi.*lambda_2_ppp*(P_2/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_2/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_1_d*(P_1/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_1/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_3*(P_3/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_3/P_i)^(1/4-1)*x)));
M_xy1ppp = @(x,s) (1- 1./(1+s*x.^(-alpha))).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) M_xy1ppp(x,s)/s, 0, Inf,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_1_2_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_2_ppp * x./p_tasso2 .* exp(-pi.*(lambda_1.*(P_1/P_i).^(2/alpha)+lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2);
U_1_2_ppp = integral(@(x) U_1_2_f_ppp(x), 0, Inf, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
%---------------------------------------------------------------------------------------------------
% to 3
P_i = P_3;
M_y1ppp = @(x,s) exp(-2*pi.*lambda_2_ppp*(P_2/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_2/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_1_d*(P_1/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_1/P_i)^(1/4-1)*x))...
                     -2*pi.*lambda_3*(P_3/P_i)^((alpha^2-alpha+2)/alpha^2)*x^(2/alpha-1)/(alpha*(1-2/alpha))*hypergeom([1,1/2],3/2,-s/((P_3/P_i)^(1/4-1)*x)));
M_xy1ppp = @(x,s) (1- 1./(1+s*x.^(-alpha))).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) M_xy1ppp(x,s)/s, 0, Inf,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_1_3_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_3 * x./p_tasso3 .* exp(-pi.*(lambda_1.*(P_1/P_i).^(2/alpha)+lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2);
U_1_3_ppp = integral(@(x) U_1_3_f_ppp(x), 0, Inf, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);

total_U_1_ppp = real(U_1_1_ppp + U_1_2_ppp + U_1_3_ppp);
fprintf('(PPP) Average Ergodic Rate [Case 1]: %.11f \n', total_U_1_ppp)
plot(a,total_U_1_ppp)


      
%%%%%%%%%%
% Case 2 %
%%%%%%%%%%
a_ = 0.0001;
M_y1ppp = @(x,s) exp(-2*pi.*lambda_2_ppp.*1./alpha*x.^(2./alpha-1)/((s*P_2).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_2/x)-...
                    2*pi.*lambda_1_d.*1./alpha*a_.^(2./alpha-1)/((s*P_1).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_1/a_)-...
                    2*pi.*lambda_3.*1./alpha*x.^(2./alpha-1)/((s*P_3).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_3/x));
   
%---------------------------------------------------------------------------------------------------
% to 1
% Null
%---------------------------------------------------------------------------------------------------
% to 2
P_i = P_2;
M_xy1ppp = @(x,s) 1./(1+s*P_i*x.^(-alpha)).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) (M_y1ppp(x,s) - M_xy1ppp(x,s))/s, 0, 1000,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_2_2_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_2_ppp * x./p_oasso23 .* exp(-pi.*(lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2);
U_2_2_ppp = integral(@(x) U_2_2_f_ppp(x), 0, 1000, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
%---------------------------------------------------------------------------------------------------
% to 3
P_i = P_3;
M_xy1ppp = @(x,s) 1./(1+s*P_i*x.^(-alpha)).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) (M_y1ppp(x,s) - M_xy1ppp(x,s))/s, 0, 1000,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_2_3_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_3 * x./p_oasso32 .* exp(-pi.*(lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2);
U_2_3_ppp = integral(@(x) U_2_3_f_ppp(x), 0, 1000, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);

total_U_2_ppp = 1.443*20.*real(U_2_2_ppp + U_2_3_ppp)/2;
fprintf('(PPP) Average Ergodic Rate [Case 2]: %.11f \n', total_U_2_ppp)


%%%%%%%%%%
% Case 3 %
%%%%%%%%%%
a_ = 0.0001;
M_y1ppp = @(x,s) exp(-2*pi.*lambda_2_ppp.*1./alpha*x.^(2./alpha-1)/((s*P_2).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_2/x)-...
                    2*pi.*lambda_1_d.*1./alpha*a_.^(2./alpha-1)/((s*P_1).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_1/a_)-...
                    2*pi.*lambda_3.*1./alpha*x.^(2./alpha-1)/((s*P_3).^-1.*(1-2./alpha)).*hypergeom([1,1-2./alpha],2-2./alpha,s*P_3/x));

%---------------------------------------------------------------------------------------------------
% to 1
%Null
%---------------------------------------------------------------------------------------------------
% to 2
P_i = P_2;
M_xy1ppp = @(x,s) 1./(1+s*P_i*x.^(-alpha)).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) (M_y1ppp(x,s) - M_xy1ppp(x,s))/s, 0, 1000,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_3_2_f_ppp = @(x,y) SINR_ppp(x).*  2 * pi .* lambda_2_ppp * x./p_3oasso123 .* exp(-pi.*(lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2).*(1-exp(-pi.*lambda_1.*(P_1/P_i).^(2/alpha)*x^2));
U_3_2_ppp = integral(@(x) U_3_2_f_ppp(x), 0, 1000, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
%---------------------------------------------------------------------------------------------------
% to 3
P_i = P_3;
M_xy1ppp = @(x,s) 1./(1+s*P_i*x.^(-alpha)).*M_y1ppp(x,s);
SINR_ppp = @(x) integral(@(s) (M_y1ppp(x,s) - M_xy1ppp(x,s))/s, 0, 1000,'ArrayValued',true, 'RelTol', 1e-1, 'AbsTol', 1e-1);
U_3_3_f_ppp = @(x) SINR_ppp(x).*  2 * pi .* lambda_3 * x./p_3oasso132 .* exp(-pi.*(lambda_2_ppp.*(P_2/P_i).^(2/alpha)+lambda_3.*(P_3/P_i).^(2/alpha))*x^2).*(1-exp(-pi.*lambda_1.*(P_1/P_i).^(2/alpha)*x^2));
U_3_3_ppp = integral(@(x) U_3_3_f_ppp(x), 0, 1000, 'ArrayValued', true, 'RelTol', 1e-1, 'AbsTol', 1e-1);

total_U_3_ppp = 1.443*20.*real(U_3_2_ppp + U_3_3_ppp)/2;
fprintf('(PPP) Average Ergodic Rate [Case 3]: %.11f \n', total_U_3_ppp)

plot(a,total_U_1_ppp,'-o',a,total_U_2_ppp,'-^',a,total_U_3_ppp,'-^')
title('Average Ergodic Rates of Different Cases')
xlabel('\beta')
ylabel('Average ergodic rate [nats/sec/MHz]')
%---------------------------------------------------------------------------------------------------
% PPP-PPCP



% for matrix operation
function y = ricepdf(x, v, s)

s2 = s.^2; % (neater below)

try
    z = besseli(0, x .* v ./ s2);
    %if ~isinf(z)
    y = (x ./ s) .*...
        exp(-0.5 * (x.^2 + v.^2) ./ s2) .*...
        z;
        % besseli(0, ...) is the zeroth order modified Bessel function of
        % the first kind. (see help bessel)
    y(x <= 0) = 0;
    k = find(isnan(y));
    y(k) = 0;
 %   else
 %       y = pdf('Rician',x,v,s);
 %   end
catch
    error('ricepdf:InputSizeMismatch',...
        'Non-scalar arguments must match in size.');
end
end

% for float operation
function y = ricepdf_(x, v, s)

s2 = s^2; % (neater below)

try
    z = besseli(0, x * v / s2);
    %if ~isinf(z)
    y = (x / s) *...
        exp(-0.5 * (x^2 + v^2) / s2) *...
        z;
        % besseli(0, ...) is the zeroth order modified Bessel function of
        % the first kind. (see help bessel)
    y(x <= 0) = 0;
    k = find(isnan(y));
    y(k) = 0;
 %   else
 %       y = pdf('Rician',x,v,s);
 %   end
catch
    error('ricepdf:InputSizeMismatch',...
        'Non-scalar arguments must match in size.');
end
end
