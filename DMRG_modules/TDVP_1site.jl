"""
 < Description >

 [ts,M,Ovals,EE] = TDVP_1site (M,Hs,O,Nkeep,dt,tmax)

 Time-dependent variational principle (TDVP) method for simulating
 real-time evolution of matrix product state (MPS). The expectation
 values of local operator O for individual sites are evaluated for
 discrete time instances. The 1D chain system is described by the matrix
 product operator (MPO) Hamiltonian H.

 < Input >
 M : [cell] The initial state as the MPS. The length of M, i.e., numel(M),
       defines the chain length. The leg convention of M{n} is as follows:

    1      3   1      3         1        3
   ---M{1}---*---M{2}---* ... *---M{end}---
       |          |                 |
       ^2         ^2                ^2

 H : [cell] MPO description of the Hamiltonian. Each Hs{n} acts on site n,
       and is rank-4 tensor. The order of legs of Hs{n} is left-bottom-
       right-top, where bottom (top) leg is to be contracted with bra
       (ket) tensor:

       |4         |4
    1  |   3   1  |   3
   ---H{1}---*---H{2}---*--- ...
       |          |
       |2         |2

 O : [matrix] Rank-2 tensor as a local operator acting on a site. The
       expectation value of this operator at each chain site is to be
       computed; see the description of the output 'Ovals' for detail.
 Nkeep : [integer] Maximum bond dimension.
 dt : [numeric] Real time step size. Each real-time evolution by step dt
       consists of one pair of sweeps (left-to-right and right-to-left).
 tmax : [numeric] Maximum time range.

 < Output >
 ts : [numeric] Row vector of discrete time values.
 M : [cell] The final MPS after real-time evolution.
 Ovals : [matrix] Ovals(m,n) indicates the expectation value of local
       operator O (input) at the site n and time ts(m).
 EE : [matrix] EE(m,n) indicates the entanglement entropy (with base 2) of
       the MPS with respect to the bipartition at the bond between the
       sites n and n+1, after acting the m-th sweep. Note that the
       time evolution from time ts(k-1) to ts(k) consists of two sweps,
       associated with the rows EE(2*k+(-1:0),:). Since the base 2 
       is chosen, the value 1 of the entanglement entropy means one
       "ebit".

 Written by S.Lee (Jun.13,2019): Written for SoSe 2019.
 Updated by S.Lee (Jun.13,2019): Revised for SoSe 2020.
 Julia version by D.Romanin (Dec.1,2020)
"""
function TDVP_1site(M,Hs,O,Nkeep,dt,tmax)

tic = time();

Nstep = round(Int,tmax/dt,RoundUp);
N = length(M);
ts = collect(dt*(1:Nstep));

## sanity check for input
if Nstep < 1
    error("ERROR: No time evolution to be done? Check dt and tmax.");
end

if N < 2
    error("ERROR: chain is too short.");
elseif length(M) != length(Hs)
    error("ERROR: M has different lengths from that of Hs.");
elseif ndims(O) != 2
    error("ERROR: local operator O should be rank 2.");
end

for itN in (1:N)
    if size(Hs[itN],2) != size(Hs[itN],4)
        error("ERROR: The second and fourth legs of Hs[$(itN)] have different dimensions.");
    elseif size(Hs[itN],2) != size(M[itN],2)
        error("ERROR: The second legs of Hs[$(itN)] and M[$(itN)] have different dimensions.");
    end
end
### 

# show message
println("1-site TDVP : Real-time evolution with local measurements");
println("N = $(length(M)), Nkeep = $(Nkeep), dt = $(dt), tmax = $(ts[end]) ( $(Nstep) steps)");

# results
Ovals = zeros(ComplexF64,Nstep,N);
EE = zeros(2*Nstep,N);

# bring into site-canonical form with respect to site 1
M,S,flagdw = canonForm(M,1);
DS = zeros(length(S),length(S));
for i in 1:length(S)
	DS[i,i] = S[i];
end
M[1] = contract(M[1],3,[3],DS,2,[1]);

## Hamiltonian for the left/right parts of the chain
Hlr = Vector{Any}(undef, N+2);
for i in 1:length(Hlr)
        Hlr[i] =  Array{Float64,3}(undef, 0,0,0);
end
# Hlr{1} and Hlr{end} are dummies; they will be kept empty. These dummies  are introduced for convenience.

# Since M is in right-canonical form by now, Hlr{..} are the right parts of
# the Hamiltonian. That is, Hlr{n+1} is the right part of Hamiltonian which
# is obtained by contracting M(n:end) with Hs(n:end). (Note the index for
# Hlr is n+1, not n, since Hlr{1} is dummy.)
for itN in (N:-1:1)
    T = permutedims(M[itN],[3 2 1]); # permute left<->right, to make use of updateLeft
    if itN == N
        # "remove" the right leg (3rd leg) of Hs{itN}, which is dummy,
        # by permuting to the last; MATLAB automatically suppresses the
        # trailing singleton dimensions
        H2 = permutedims(Hs[itN],[2 1 4 3]); # bottom-left-top (-right)
	H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
        Hlr[itN+1] = updateLeft([],[],T,H2,3,T);
    elseif itN == 1
        Hlr[itN+1] = Array{Float64,3}(undef, 0,0,0);
    else
        # permute left<->right, to make use of updateLeft
        H2 = permutedims(Hs[itN],[3 2 1 4]); # right-bottom-left-top
        Hlr[itN+1] = updateLeft(Hlr[itN+2],3,T,H2,4,T);
    end
end

println("Time evolution: start");

for itt = (1:Nstep)
    # left -> right
    for itN in (1:(N-1))
        # time evolution of site-canonical tensor ("A tensor") M{itN}, via TDVP_1site_expHA
        Anew = TDVP_1site_expHA(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN],dt/2);

	# update M{itN} and generate Cold by using Anew, via SVD
        M[itN],S2,V2 =  svdTr(Anew,3,[1,2],Nkeep,-1); # set Stol as -1, not to truncate even zero singular values
	DS2 = zeros(length(S2),length(S2));
	for i in 1:length(S2)
		DS2[i,i] = S2[i];
	end
        Cold = contract(DS2,2,[2],V2,2,[1]);	

	# entanglement entropy
	Evec = (S2.^2).*log.(S2);
        for i in 1:length(Evec)
                if isnan(Evec[i]) || isinf(Evec[i])
                        Evec[i] = 0.0;
                end
        end
        EE[itt*2-1,itN] = (2/log(2))*sum(-Evec);
	
        # update Hlr{itN+1} in effective basis
 	if itN == 1
            # "remove" the left leg (1st leg) of Hs{itN}, which is dummy,
            # by permuting to the last; MATLAB automatically suppresses the
            # trailing singleton dimensions
            H2 = permutedims(Hs[itN],[2 3 4 1]); # bottom-right-top (-left)
            H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
            Hlr[itN+1] = updateLeft([],[],M[itN],H2,3,M[itN]);
        else
            Hlr[itN+1] = updateLeft(Hlr[itN],3,M[itN],Hs[itN],4,M[itN]);
        end

        # inverse time evolution of C tensor (Cold -> Cnew), via TDVP_1site_expHC
        Cnew = TDVP_1site_expHC(Hlr[itN+1],Hlr[itN+2],Cold,dt/2);

        # absorb Cnew into M{itN+1}
        M[itN+1] = contract(Cnew,2,[2],M[itN+1],3,[1]);
    end

    itN = N; # right end
    M[itN] = TDVP_1site_expHA(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN],dt);

   # right -> left
   for itN = ((N-1):-1:1)
       # update M{itN+1} and generate Cold via SVD
        U2,S2,M[itN+1] = svdTr(M[itN+1],3,[1],Nkeep,-1); # set Stol as -1, not to truncate even zero singular values
	DS2 = zeros(length(S2),length(S2));
        for i in 1:length(S2)
                DS2[i,i] = S2[i];
        end
        Cold = contract(U2,2,[2],DS2,2,[1]);

        # entanglement entropy
        Evec = (S2.^2).*log.(S2);
        for i in 1:length(Evec)
                if isnan(Evec[i]) || isinf(Evec[i])
                        Evec[i] = 0.0;
                end
        end
        EE[itt*2,itN] = (2/log(2))*sum(-Evec);

       # update Hlr{itN+2} in effective basis
       T = permutedims(M[itN+1],[3 2 1]); # permute left<->right, to make use of updateLeft
       if (itN+1) == N
           # "remove" the right leg (3rd leg) of Hs{itN}, which is dummy,
           # by permuting to the last; MATLAB automatically suppresses the
           # trailing singleton dimensions
            H2 = permutedims(Hs[itN+1],[2 1 4 3]); # bottom-left-top (-right)
            H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
            Hlr[itN+2] = updateLeft([],[],T,H2,3,T);
       else
           # permute left<->right for Hs{itN} as well, to make use of updateLeft
           H2 = permutedims(Hs[itN+1],[3 2 1 4]); # right-bottom-left-top
           Hlr[itN+2] = updateLeft(Hlr[itN+3],3,T,H2,4,T);
       end

        # inverse time evolution of C tensor (Cold -> Cnew), via TDVP_1site_expHC
        Cnew = TDVP_1site_expHC(Hlr[itN+1],Hlr[itN+2],Cold,dt/2);

        # absorb Cnew into M{itN}
        M[itN] = contract(M[itN],3,[3],Cnew,2,[1]);

        # time evolution of site-canonical tensor ("A tensor") M{itN}, via TDVP_1site_expHA
        M[itN] = TDVP_1site_expHA(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN],dt/2);
   end

    # Measurement of local operators O; currently M is in site-canonical
    # with respect to site 1
    MM = []; # contraction of bra/ket tensors from left
    for itN in (1:N)
        Ovals[itt,itN] = tr(updateLeft(MM,2,M[itN],O,2,M[itN]));
        MM = updateLeft(MM,2,M[itN],[],[],M[itN]);
    end

    if (mod(itt,round(Nstep/10)) == 0) || (itt == Nstep)
	print_time();
	println(" | # $(itt)/$(Nstep) : t = $(ts[itt])/$(ts[end])")
    end
end

toc = time()-tic;

println("Elapsed time: $(toc) s.")

return ts,M,Ovals,EE;

end

"""
 Time evolution for "A tensor" Aold by time step dt, by using Hlr tensors
 that are the Hamiltonian in effective basis. Anew is the result after
 time evolution.
 This subfunction is adapted from the Lanczos routine 'DMRG/eigs_1site.m'.
 The difference from 'eigs_1site' is that it considers the time evolution,
 not the ground state, and does not consider the orthonormal state.
"""
function TDVP_1site_expHA(Hleft,Hloc,Hright,Aold,dt)

# default parameters
N = 5;
minH = 1e-10;

Asz = (size(Aold,1),size(Aold,2),size(Aold,3)); # size of ket tensor
Akr = complex(zeros(length(Aold),N+1)); # Krylov vectors (vectorized tensors)
Akr[:,1] = (reshape(Aold[:],(length(Aold))))/norm(reshape(Aold[:],(length(Aold)))); # normalize Aold

# In the Krylov basis, the Hamiltonian becomes tridiagonal
ff = complex(zeros(N,1)); # 1st diagonal
gg = complex(zeros(N+1,1)); # main diagonal

for itN in (1:(N+1))
    # contract Hamiltonian with ket tensor
    Atmp = TDVP_1site_HA(Hleft,Hloc,Hright,reshape(Akr[:,itN],Asz));
    Atmp = Atmp[:]; # vectorize

    gg[itN] = Akr[:,itN]'*Atmp; # diagonal element, "on-site energy"

    if itN < (N+1)
        # orthogonalize Atmp w.r.t. the previous ket tensors
        Atmp = Atmp - Akr[:,1:itN]*(Akr[:,1:itN]'*Atmp);
        Atmp = Atmp - Akr[:,1:itN]*(Akr[:,1:itN]'*Atmp); # twice, to reduce numerical noise

        # norm
        ff[itN] = norm(Atmp);

        if abs(ff[itN]) > minH
            Akr[:,itN+1] = Atmp/ff[itN];
        else
            # stop iteration; truncate ff, gg
            ff = ff[1:(itN-1)];
            gg = gg[1:itN];
            Akr = Akr[:,(1:itN)];
            break
        end
    end
end

# Hamiltonian in the Krylov basis
Hkr = complex(zeros(length(gg),length(gg)));
for i in 1:length(ff)
        Hkr[i,i] = gg[i]/2.0;
        Hkr[i,i+1] = ff[i];
        if i == length(ff)
                Hkr[i+1,i+1] = gg[i+1]/2.0;
        end
end
Hkr = Hkr + Hkr';
Ekr,Vkr = eigen((Hkr+Hkr')/2.0);

Anew = Akr*(Vkr*(Diagonal(exp.((-1.0*im*dt)*Ekr))*reshape(Vkr[1,:]',(length(Vkr[1,:])))));
Anew = Anew/norm(Anew); # normalize
Anew = reshape(Anew,Asz); # reshape back to rank-3 tensor

return Anew;

end

function TDVP_1site_HA(Hleft,Hloc,Hright,Ain)

# set empty tensors as 1, for convenience
if isempty(Hleft)
    Hleft = getIdentity(Ain,1,Hloc,1,[3 2 1]);
end
if isempty(Hright)
    Hright = getIdentity(Ain,3,Hloc,3,[1 3 2]);
end

Aout = contract(Hleft,3,[3],Ain,3,[1]);
Aout = contract(Aout,4,[2 3],Hloc,4,[1 4]);
Aout = contract(Aout,4,[2 4],Hright,3,[3 2]);

return Aout;

end

"""
 Time evolution for "C tensor" Aold by time step -dt, by using Hlr tensors
 that are the Hamiltonian in effective basis. Cnew is the result after
 time evolution.
 This subfunction is adapted from the subfunction 'TDVP_1site_expHA'
 above. The differences here are that the ket tensor is rank-2; that there
 is no the local Hamiltonian 'Hloc'; and that the time evolution is in an
 inverse direction.
"""
function TDVP_1site_expHC(Hleft,Hright,Aold,dt)

# default parameters
N = 5;
minH = 1e-10;

Asz = (size(Aold,1),size(Aold,2)); # size of ket tensor
Akr = complex(zeros(length(Aold),N+1)); # Krylov vectors (vectorized tensors)
Akr[:,1] = reshape(Aold[:],(length(Aold)))/norm(reshape(Aold[:],(length(Aold)))); # normalize Aold

# In the Krylov basis, the Hamiltonian becomes tridiagonal
ff = complex(zeros(N,1)); # 1st diagonal
gg = complex(zeros(N+1,1)); # main diagonal

for itN = (1:(N+1))
    # contract Hamiltonian with ket tensor
    Atmp = TDVP_1site_HC(Hleft,Hright,reshape(Akr[:,itN],Asz));
    Atmp = Atmp[:]; # vectorize

    gg[itN] = Akr[:,itN]'*Atmp; # diagonal element, "on-site energy"

    if itN < (N+1)
        # orthogonalize Atmp w.r.t. the previous ket tensors
        Atmp = Atmp - Akr[:,1:itN]*(Akr[:,1:itN]'*Atmp);
        Atmp = Atmp - Akr[:,1:itN]*(Akr[:,1:itN]'*Atmp); # twice, to reduce numerical noise

        # norm
        ff[itN] = norm(Atmp);

        if abs(ff[itN]) > minH
            Akr[:,itN+1] = Atmp/ff[itN];
        else
            # stop iteration; truncate ff, gg
            ff = ff[1:(itN-1)];
            gg = gg[1:itN];
            Akr = Akr[:,(1:itN)];
            break
        end
    end
end

# Hamiltonian in the Krylov basis
Hkr = complex(zeros(length(gg),length(gg)));
for i in 1:length(ff)
        Hkr[i,i] = gg[i]/2.0;
        Hkr[i,i+1] = ff[i];
        if i == length(ff)
                Hkr[i+1,i+1] = gg[i+1]/2.0;
        end
end
Hkr = Hkr + Hkr';
Ekr,Vkr = eigen((Hkr+Hkr')/2.0);

Anew = Akr*(Vkr*(Diagonal(exp.((+1.0*im*dt)*Ekr))*reshape(Vkr[1,:]',(length(Vkr[1,:])))));
Anew = Anew/norm(Anew); # normalize
Anew = reshape(Anew,Asz); # reshape back to rank-3 tensor

return Anew;

end

function TDVP_1site_HC(Hleft,Hright,Ain)

# set empty tensors as 1, for convenience
if isempty(Hleft)
    Hleft = getIdentity(Ain,1,Hright,2,[3 2 1]);
end
if isempty(Hright)
    Hright = getIdentity(Ain,3,Hleft,2,[3 2 1]);
end

Aout = contract(Hleft,3,[3],Ain,2,[1]);
Aout = contract(Aout,3,[2 3],Hright,3,[2 3]);

return Aout;

end
