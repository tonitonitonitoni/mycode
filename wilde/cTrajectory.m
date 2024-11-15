function zDe=cTrajectory(t)
% continuous trajectory for FFSM
% one joint at a time accelerates to 0.1 rad/s for 10 s, then rests for 2 s, 
% rotates at −0.1 rad/s for 20 s, rests again for 2 s, 
% then rotates back to the initial condition.
n=4;
if t>4*44-1
    qS=zeros(n,1);
    qSdot=zeros(n,1);
else  
    t_t=mod(t,44);
    j=floorDiv(t,44)+1;
    qS=zeros(n,1);
    qSdot=zeros(n,1);
    
        if t_t<=10
            qSdot(j)=0.1;
            qS(j)=0.1*t_t;
        elseif t_t<=12
            qSdot(j)=0.0;
            qS(j)=1.0;
        elseif t_t<=32
            qSdot(j)=-0.1;
            qS(j)=1+qSdot(j)*(t_t-12);
        elseif t_t<=34
            qSdot(j)=0.0;
            qS(j)=-1.0;
        else
            qSdot(j)=0.1;
            qS(j)=-1.0+qSdot(j)*(t_t-34);
        end
end
    zDe=[qS;qSdot];
end