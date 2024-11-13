function x=rot321inv(R)
a=R(3,1);
theta=atan2(-a, sqrt(1-a^2));
phi=atan2(R(3,2),R(3,3));
psi=atan2(R(2,1),R(1,1));
x=[phi;theta;psi];
end
