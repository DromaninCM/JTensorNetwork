"""
 < Description >

 [ts,M,Ovals,EE,dw] = tDMRG (M,Hs,O,Nkeep,dt,tmax)

 Real time evolution by using time-dependent DMRG (tDMRG) for simulating
 real-time evolution of matrix product state (MPS). The expectation
 values of local operator O for individual sites are evaluated for
 discrete time instances. The 1D chain system is described by the
 Hamiltonian H. Here we use the 2nd order Trotter step exp(-dt/2*Hodd) *
 exp(-dt*Heven) * exp(-dt/2*Hodd). After acting each exp(-t*H) type
 operator, the bonds are truncated such that the largest singular values
 are kept.

 < Input >
 M : [cell] The initial state as the MPS. The length of M, i.e., numel(M),
       defines the chain length. The leg convention of M{n} is as follows:

    1      3   1      3         1        3
   ---M{1}---*---M{2}---* ... *---M{end}---
       |          |                 |
       ^2         ^2                ^2

 H : [cell] Hamiltonian. Each cell element H{n} describes the two-site
       interaction between site n and n+1. Thus, H(1:2:end) acts on odd
       bonds; H(2:2:end) on even bonds. It should satisfy numel(M) ==
       numel(H) + 1.
       The leg convention of H{n} are as follows:

       2      4     [legs 1 and 2 are for site n;
       |      |       legs 3 and 4 are for site n+1]
      [  H{n}  ]
       |      |
       1      3

 O : [matrix] Rank-2 tensor as a local operator acting on a site.
 Nkeep : [integer] Maximum bond dimension.
 dt : [numeric] Real time step size. Each real-time evolution by step dt
       consists of three Trotter steps, exp(-dt/2*Hodd) * exp(-dt*Heven) *
       exp(-dt/2*Hodd).
 tmax : [numeric] Maximum time range.

 < Output >
 ts : [numeric] Row vector of discrete time values.
 M : [cell] The final MPS after real-time evolution.
 Ovals : [matrix] Ovals(m,n) indicates the expectation value of local
       operator O (input) at the site n and time ts(m).
 EE : [matrix] EE(m,n) indicates the entanglement entropy (with base 2) of
       the MPS with respect to the bipartition at the bond between the
       sites n and n+1, after acting the m-th Trotter step. Note that the
       time evolution from time ts(k-1) to ts(k) consists of three Trotter
       steps, associated with the rows EE(3*k+(-2:0),:). Since the base 2
       is chosen, the value 1 of the entanglement entropy means one
       "ebit".
 dw : [matrix] Discarded weights (i.e., the sum of the squares of the
       discarded singular values) after each Trotter step. dw(m,n)
       corresponds to the same bond and Trotter step associated with
       EE(m,n).

 Written by S.Lee (Jun.19,2017); updated by S.Lee (Jun.22,2017)
 Updated by S.Lee (Jun.07,2019): Revised for SoSe 2019.
"""
function tDMRG(M,Hs,O,Nkeep,dt,tmax)

tic = time();

### check the integrity of input
if length(M) != (length(Hs)+1)
    error("ERROR: it should be length(M) == (length(H)+1)");
elseif ndims(O) != 2
    error("ERROR: local operator O should be rank 2.");
end

for itN in (1:length(Hs))
    if size(Hs[itN],1) != size(Hs[itN],2)
        error("ERROR: The first and third legs of Hs[$(itN)] have different dimensions.");
    elseif size(Hs[itN],3) != size(Hs[itN],4)
        error("ERROR: The third and fourth legs of Hs[$(itN)] have different dimensions.");
    elseif size(Hs[itN],1) != size(M[itN],2)
        error("ERROR: The first leg of Hs[$(itN)] and the second leg of M[$(itN)] have different dimensions.");
    elseif size(Hs[itN],3) != size(M[itN+1],2)
        error("ERROR: The third leg of Hs[$(itN)] and the second leg of M[$(itN+1)] have different dimensions.");
    end
end
### 

Nstep = round(Int,tmax/dt,RoundUp);

# results
ts = collect(dt*(1:Nstep));
Ovals = zeros(ComplexF64,Nstep,length(M));
EE = zeros(3*Nstep,length(M)-1);
dw = zeros(size(EE));

# show information
println("tDMRG: Real-time evolution with local measurements");
println("N = $(length(M)), Nkeep = $(Nkeep), dt = $(dt), tmax = $(ts[end]) ( $(Nstep) steps)");

# generate the unitray operator exp(-it*H) for each two-site pairs
expH = Vector{Array{ComplexF64,4}}(undef, length(Hs));
for i in 1:length(expH)
       expH[i] = Array{ComplexF64}(undef,0,0,0,0);
end
for it1 in (1:length(Hs))
    if !isempty(Hs[it1])
        sdim = [size(M[it1],2),size(M[it1+1],2)];
        Htmp = permutedims(Hs[it1],[1 3 2 4]);
        Htmp = reshape(Htmp,(sdim[1]*sdim[2],sdim[1]*sdim[2]));
        if mod(it1,2) == 1
            ttmp = dt/2; # half time step for odd bonds, as the time evolution steps for odd bonds will happen twice
        else
            ttmp = dt;
        end
        DH,VH = eigen(Htmp);
	BH = zeros(ComplexF64,length(DH),length(DH))
	for i in 1:length(DH)
        	BH[i,i] = exp((-1*im*ttmp)*DH[i]);
    	end
        eH = VH*BH*VH';
        expH[it1] = reshape(eH,Tuple(append!(sdim,sdim)));
    end
end

print_time();
println(" | Transform the MPS into right-canonical form");
M,flagS,flagdw = canonForm(M,0);

print_time();
println(" | Trotter steps: start");

for it1 in (1:3*Nstep)
# Here we use the 2nd order Trotter step exp(-dt/2*Hodd) * exp(-dt*Heven) *
# exp(-dt/2*Hodd). That is, for the case mod(it1,3) == 2, we act the
# unitary on even bonds. Otherwise, on odd bonds.
    expHtmp = Vector{Array{ComplexF64,4}}(undef, length(Hs));
    for i in 1:length(expH)
       expHtmp[i] = Array{ComplexF64}(undef,0,0,0,0);
    end
    if mod(it1,3) == 2 				# even bonds
        expHtmp[2:2:end] = expH[2:2:end];
    else 					# odd bonds
        expHtmp[1:2:end] = expH[1:2:end];
    end

    # call local function tDMRG_1sweep
    M,EE1,dw1 = tDMRG_1sweep(M,expHtmp,Nkeep,mod(it1,2));

    # update the rows of entanglement entropy (EE) and discarded weights (dw)
    EE[it1,:] = EE1;
    dw[it1,:] = dw1;

    # evaluate local expectation values
    if mod(it1,3) == 0
        Ovals[Int(it1/3),:] = tDMRG_expVal(M,O,mod(it1,2));
    end

    if (mod(it1,round(3*Nstep/10)) == 0) || (it1 == (3*Nstep))
	print_time();
	println(" | #$(Int(it1/3))/$(Nstep) : t = $(ts[Int(it1/3)])/$(ts[end])");
    end
end

toc = time() - tic;

println("Elapsed time: $(toc) s");

return ts,M,Ovals,EE,dw;

end

"""
 Apply exp(-it*H), which is an array of two-site gates acting on either
 even or odd bonds, and then truncate bonds by using SVD. After applying
 this function, left-canonical state becomes right-canonical, and vice
 versa.

 < Input >
 M : [cell] Input MPS.
 expH : [cell] exp(-i*H*T) unitary operators for each bond. The length
       should satisfy numel(expH) == numel(M)-1. And the every first (or
       second) elements should be empty, since we act either even or odd
       bonds at once.
 Nkeep : [numeric] Maximum bond dimension.
 isright : [logical] If true, we take left-to-right sweep. Otherwise, take
       right-to-left sweep.
 
 < Output >
 M : [cell] MPS after applying exp(-it*H) and truncating bonds.
 EE : [numeric vector] Entanglement entropy at each bond.
 dw : [numeric vector] Discarded weights when truncating the bond
       dimensions.
"""
function tDMRG_1sweep(M,expH,Nkeep,isright)

N = length(M);
EE = zeros(1,N-1);
dw = zeros(1,N-1);

if isright == 1 # left -> right
    for it = (1:N-1)
        # contract M[it] and M[it+1] with expH[it]
        T = contract(M[it],3,[3],M[it+1],3,[1]);
        if !isempty(expH[it])
            T = contract(expH[it],4,[3 4],T,4,[2 3],[3 1 2 4]);
        end
        # SVD via svdTr
        M[it],S,V,dw[it] = svdTr_dw(T,4,[1,2],Nkeep,[]);
        # normalize the singular values, to normalize the norm of MPS
        S = S/norm(S);
        # compiute entanglement entropy of base 2. Be aware of zero
        # singular values!
        Evec = (S.^2).*log.(S);
	for i in 1:length(Evec)
		if isnan(Evec[i]) || isinf(Evec[i])
			Evec[i] = 0.0;
		end
	end
        EE[it] = (2/log(2))*sum(-Evec);
        # update M[it+1]
	DS = zeros(length(S),length(S))
	for i in 1:length(S)
		DS[i,i] = S[i];
	end
        M[it+1] = contract(DS,2,[2],V,3,[1]);
    end
    M[end] = M[end]/norm(M[end][:]); # to normalize the norm of MPS
else # right -> left
    for it in (N-1:-1:1)
        # contract M{it} and M{it+1} with expH{it}
        T = contract(M[it],3,[3],M[it+1],3,[1]);
        if !isempty(expH[it])
            T = contract(expH[it],4,[3 4],T,4,[2 3],[3 1 2 4]);
        end
        # SVD via svdTr
        U,S,M[it+1],dw[it] = svdTr_dw(T,4,[1,2],Nkeep,[]);
        # normalize the singular values, to normalize the norm of MPS
        S = S/norm(S);
        # compute entanglement entropy of base 2. Be aware of zero
        # singular values!
        Evec = (S.^2).*log.(S);
	for i in 1:length(Evec)
		if isnan(Evec[i]) || isinf(Evec[i])
			Evec[i] = 0.0;
		end
	end
        EE[it] = (2/log(2))*sum(-Evec);
        # update M{it}
        DS = zeros(length(S),length(S))
        for i in 1:length(S)
                DS[i,i] = S[i];
        end
        M[it] = contract(U,3,[3],DS,2,[1]);
    end
    M[1] = M[1]/norm(M[1][:]); # to normalize the norm of MPS
end

return M,EE,dw;

end

"""
 Expectation values of local operator O (acting on only one site) for
 given MPS.

 < Input >
 M : [cell] Input MPS.
 O : [matrix] Rank-2 operator acting on one site.
 isleft : [logical] If true, it means that the MPS M is in left-canonical
       form. Otherwise, right-canonical form.

 < Output >
 Ovals : [vector] Expectation value of operator O. It will substitute the
       rows of Ovals in the main function tDMRG.
"""
function tDMRG_expVal(M,O,isleft)

N = length(M);
Ovals = zeros(ComplexF64,1,N);

MM = []; 						# contraction of bra/ket tensors
if isleft == 1 						# left-normalized
    for itN in (N:-1:1)
        T = permutedims(M[itN],[3 2 1]); 		# permute left<->right to make use of updateLeft
        Ovals[itN] = tr(updateLeft(MM,2,T,O,2,T));
        MM = updateLeft(MM,2,T,[],[],T);
    end
else 							# right-normalized
    for itN in (1:N)
        Ovals[itN] = tr(updateLeft(MM,2,M[itN],O,2,M[itN]));
        MM = updateLeft(MM,2,M[itN],[],[],M[itN]);
    end
end

return Ovals;

end
