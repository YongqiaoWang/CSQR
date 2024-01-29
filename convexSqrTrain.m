function [vAlpha,mBeta,vFitY]=convexSqrTrain(mTrainX,vTrainY,tau)

[n,p]=size(mTrainX);
ntau=ceil(n*tau);

if ntau<n
    vCoefT1(ntau,1)=ntau/n-tau
    vCoefT3(ntau,1)=log(1-tau)-log(1-ntau/n)
    for k=(ntau+1):(n-1)
        vCoefT1(k,1)=1/n;
        vCoefT3(k,1)=log(1-(k-1)/n)-log(1-k/n);
    end
    vCoefT1=vCoefT1(ntau:(n-1))/(1-tau);
    vCoefT3=vCoefT3(ntau:(n-1))/(n*(1-tau));

    cvx_begin
        variable z(n-ntau,1)
        variable E(n-ntau,n) nonnegative
        variable a(n,1)
        variable B(n,p) nonnegative
        variable r(n,1)
        variable w
        minimize(z'*vCoefT1+w/(n*(1-tau))+vCoefT3'*sum(E,2)-sum(r)/n)
        subject to
            for i=1:n
                r(i)==vTrainY(i)-a(i)-B(i,:)*mTrainX(i,:)'
            end
            r<=w*ones(n,1)
            for i=1:n
                for k=ntau:(n-1)
                    E(k-ntau+1,i)>=r(i)-z(k-ntau+1)
                end
            end
           for i=1:n
               for j=1:n
                   a(i)+B(i,:)*mTrainX(i,:)'<=a(j)+B(j,:)*mTrainX(i,:)'
               end
           end
    cvx_end
else
    cvx_begin
        variable a(n,1)
        variable B(n,p) nonnegative
        variable r(n,1)
        variable w
        minimize(w-sum(r)/n)
        subject to
            for i=1:n
                r(i)==vTrainY(i)-a(i)-B(i,:)*mTrainX(i,:)'
            end
            r<=w*ones(n,1)
            for i=1:n
               for j=1:n
                   a(i)+B(i,:)*mTrainX(i,:)'<=a(j)+B(j,:)*mTrainX(i,:)'
               end
            end
     cvx_end
end


cvx_begin
    variable a0
    variable e(n) nonnegative
    minimize(a0+sum(e)/(n*(1-tau)))
    subject to
        for i=1:n
            e(i)>=vTrainY(i)-a(i)-B(i,:)*mTrainX(i,:)'-a0
        end
cvx_end

vAlpha=cvx_optval+a;
mBeta=B;
for i=1:n
    vFitY(i,1)=vAlpha(i)+mBeta(i,:)*mTrainX(i,:)';
end



