function z = zF(phi,psi,theta)
%zF
%    Z = zF(PHI,PSI,THETA)

%    This function was generated by the Symbolic Math Toolbox version 24.1.
%    10-Nov-2024 12:48:50

t2 = cos(phi);
t3 = cos(psi);
t4 = cos(theta);
t5 = sin(phi);
t6 = sin(psi);
t7 = sin(theta);
t8 = t2.*t4;
t9 = t3.*t5;
t10 = t5.*t6;
t12 = t2.*t3.*t7;
t13 = t2.*t6.*t7;
t11 = -t9;
t14 = t10+t12;
t15 = t11+t13;
z = reshape([t14,t15,t8,t14,t15,t8,t14,t15,t8,t14,t15,t8,t14,t15,t8],[3,5]);
end