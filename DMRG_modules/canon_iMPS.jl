"""
 < Description >

 [Lambda,Gamma] = canon_iMPS (Lambda,Gamma)

 Orthogonalize Vidal's Gamma-Lambda representation of infinite MPS,
 following the method given in [R. Orus & G. Vidal, Phys. Rev. B 78,
 155117 (2008)]. Here the goal of the orthogonalization is to make the ket
 tensors of Lambda*Gamma type (Gamma*Lambda type) be left-normalized
 (right-normalized), i.e., to bring them into canonical forms.

 < Input >
 Lambda : [cell] Each cell contains the column vector of the singular
       values at each bond. Number of cells, numel(Lambda), means the size
       of the unit cell.
 Gamma : [cell] Each cell contains a rank-3 tensor associated with each
       site within an unit cell. Number of cells, numel(Gamma), means the
       size of the unit cell, and needs to be the same as numel(Lambda).
       The ket tensor for the unit cell is represented by:

 ->-diag(Lambda{end})->-*->-Gamma{1}->-* ... *->-diag(Lambda{end-1})->-*->-Gamma{end}->-*->-diag(Lambda{end})->-  
  1                   2   1    ^     3         1                     2   1    ^       3   1                   2 
                               |2                                             |2

       The Lambda's and Gamma's can be given as random tensors. After the
       imaginary time evoltion, the ket tensor becomes left-normalized (up
       to some numerical error).

 < Output >
 Lambda, Gamma : [cell] Cell arrays of Lambda and Gamma tensors,
       repectively, after the orthogonalization.

 Written by S.Lee (Jun.16,2017); updated by S.Lee (Jun.19,2017)
 Updated by S.Lee (Jun.01,2019): Revised for Sose 2019.
 Updated by S.Lee (Jun.09,2020): Typo fixed.
 Julia version by D.Romanin (Nov.27,2020)
"""
function canon_iMPS(Lambda,Gamma)

Lambda = Lambda[:];
Gamma = Gamma[:];
n = length(Lambda); # number of sites in the unit cell

### check the integrity of input
for it in 1:n
    if length(Lambda[mod(it,n)+1]) != size(Gamma[it],1)
        error("ERROR: Dimensions for Lambda[$(mod(it,n)+1)] and Gamma[$(it)] do not match.");
    elseif length(Lambda[it]) != size(Gamma[it],3)
        error("ERROR: Dimensions for Lambda[$(it)] and Gamma[$(it)] do not match.");
    end
end
### 

## "Coarse grain" the tensors: contract the tensors for the unit cell
## altogether. Then the coarse-grained tensor will be orthogonalized.
# Gamma{1}*Lambda{1}* ... * Gamma{end}
T = Gamma[1];
for it in 2:n
    DL = zeros(length(Lambda[it-1]),length(Lambda[it-1]));
    for i  in 1:length(Lambda[it-1])
	DL[i,i] = Lambda[it-1][i];
    end
    T = contract(T,it+1,[it+1],DL,2,[1]);
    T = contract(T,it+1,[it+1],Gamma[it],3,[1]);
end

# ket tensor to compute transfer operator from right
DL = zeros(length(Lambda[n]),length(Lambda[n]));
for i  in 1:length(Lambda[n])
	DL[i,i] = Lambda[n][i];
end
TR = contract(T,n+2,[n+2],DL,2,[1]);
# find the dominant eigenvector for the transfer operator from right
XR = canonIMPS_domVec(TR,n+2,false);

# ket tensor to compute transfer operator from left
TL = contract(DL,2,[2],T,n+2,[1]);
# find the dominant eigenvector for the transfer operator from left
XL = canonIMPS_domVec(TL,n+2,true);

# do SVD in Fig. 2(ii) of [R. Orus & G. Vidal, Phys. Rev. B 78, 155117 (2008)]
U,SX,V = svdTr(XL*DL*XR,2,[1],[],[]);

# orthogonalize the coarse-grained tensor
T = contract(V/XR,2,[2],T,n+2,[1]);
T = contract(T,n+2,[n+2],XL\U,2,[1]);

# result
Lambda = Vector{Union{Nothing, Array{Float64,1}}}(nothing, n);
Gamma = Vector{Union{Nothing, Array{Float64,3}}}(nothing, n);
Lambda[n] = SX/norm(SX);

if n == 1
    # there is only one site in the unit cell; the job is done
    Gamma[1] = T;
else
    # decompose the orthogonalized coarse-grained tensor into the tensors
    # for individual sites

    # contract singular value tensors to the left and right ends, before
    # doing SVD.
    DL = zeros(length(Lambda[n]),length(Lambda[n]));
    for i  in 1:length(Lambda[n])
        DL[i,i] = Lambda[n][i];
    end
    T = contract(DL,2,[2],T,n+2,[1]);
    T = contract(T,n+2,[n+2],DL,2,[1]);

    V2 = [];
    for it in 1:n-1
        if it > 1
            # contract singular value tensor to the left, since the
            # singular value tensor contracted to the left before is
            # factored out by SVD
	    DL = zeros(length(Lambda[it-1]),length(Lambda[it-1]));
            for i  in 1:length(Lambda[it-1])
        	DL[i,i] = Lambda[it-1][i];
            end
            T = contract(DL,2,[2],V2,n+2-(it-1),[1]);
        end
        U2,S2,V2 = svdTr(T,n+2-(it-1),[1,2],[],[]);
        Lambda[it] = S2/norm(S2);
	DL = zeros(length(Lambda[mod(it-2,n)+1]),length(Lambda[mod(it-2,n)+1]));
        for i  in 1:length(Lambda[mod(it-2,n)+1])
		DL[i,i] = 1/Lambda[mod(it-2,n)+1][i];
        end
        Gamma[it] = contract(DL,2,[2],U2,3,[1]);
    end

    DL = zeros(length(Lambda[n]),length(Lambda[n]));
    for i  in 1:length(Lambda[n])
        DL[i,i] = 1/Lambda[n][i];
    end
    Gamma[end] = contract(V2,3,[3],DL,2,[1]);
end

return Lambda,Gamma;

end

"""
% Find the tensor X where X*X' originates from the dominant right
% eigenvector of the transfer operator which is
%
%  1                 n+2
%  -->-[     T     ]->--
%      2|   ...   |n+1
%       ^         ^
%       *   ...   *
%       ^         ^
%      2|   ...   |n+1
%  --<-[     T'    ]-<--
%  1                 n+2
%
% isconj == 1 if transfer operator from left, == 0 from right
"""
function canonIMPS_domVec(T,rankT,isconj)

D = size(T,1);
idT = collect(2:rankT-1);

T = contract(T,rankT,idT,conj(T),rankT,idT,[1 3 2 4]);
T = reshape(T,(D*D,D*D));

if isconj
	T = copy(T');
end

DE,V = eigen(T);
DE = real(DE);
V = real(V);
ids = sortperm(DE,rev=true);
V = reshape(V[:,ids[1]],(D,D));
V = (V+V')/2; # Hermitianize for numerical stability
D1,V1 = eigen(V);

# V is equivalent up to overall sign; if the eigenvalues are mostly
# negative, take the overall sign flip
if sum(D1) < 0
    D1 = -D1;
end

# remove negative eigenvalues, which come from numerical noise
oks = collect(1:length(D1));
idF = [];
for i in 1:length(D1)
	if D1[i]<=0
		idF = append!(idF,i);
	end
end
filter!(i->(i ∉ idF),oks)
D1 = D1[oks];
V1 = V1[:,oks];

DL = zeros(length(D1),length(D1));
for i in 1:length(D1)
	DL[i,i] = sqrt(D1[i]);
end
X = V1*DL;
if isconj
	 X = copy(X');
end

return X

end
