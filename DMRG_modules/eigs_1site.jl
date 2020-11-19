"""
 < Description >

 [Aeff,Eeff] = eigs_1site (Hleft,Hloc,Hright,Ainit [, option])

 Obtain the ground state and its energy for the effective Hamiltonian for
 the site-canonical MPS, by using the Lanczos method.

 < Input >
 Hleft, Hloc, Hright: [tensors] The Hamiltonian for the left, site, and
       right parts of the chain. They form the effective Hamiltonian in
       the site-canonical basis.
 Ainit : [tensor] Ket tensor at a lattice site. It becomes an initial
       vector for the Lanczos method.

 The input tensors can be visualized as follows:
 (numbers are the order of legs, * means contraction)

                
      1 -->-[ Ainit or Aorth ]-<-- 3  
                     |                  
                     ^ 2                          
                     |                            
                                                  
                                                  
     /--->- 3        | 4        3 -<---/          
     |               ^                 |   
     |    2     1    |    3     2      |   
   Hleft-->- * -->- Hloc->-- * ->-- Hright 
     |               |                 |   
     |               ^                 |   
     /---<- 1        | 2        1 ->---/   

 < Option >
 'Aorth', .. : [tensor] If not given, the ground state of the whole space
       spanned by Hleft, Hloc, and Hright will be searched via the 
       Lanczos method. If specified, the search is contrained to the
       subspace orthogonal to Aorth.
 'N', .. : [numeric] Maximum number of Lanczos vectors (in addition to
       those given by Ainit and Aorth) to be considered for the Krylov
       subspace.
       (Default: 5)
 'minH', .. : [numeric] Minimum absolute value of the 1st diagonal (i.e.,
       superdiagonal) element of the Hamiltonian in the Krylov subspace.
       If a 1st-diagonal element whose absolute value is smaller than minH
       is encountered, the iteration stops. Then the ground-state vector
       and energy are obtained from the tridiagonal matrix constructed so
       far.
       (Default: 1e-10)

 < Output >
 Aeff : [tensor] A ket tensor as the ground state of the effective
       Hamiltonian.
 Eeff : [numeric] The energy eigenvalue corresponding to Aeff.
 
 Written by S.Lee (May 31,2017)
 Documentation updated by S.Lee (Jun.8,2017)
 Updated by S.Lee (May 28,2019): Revised for SoSe 2019.
 Updated by S.Lee (May 23,2020): Revised for SoSe 2020.
 Updated by S.Lee (Jun.08,2020): Typo fixed.
 Julia version by D. Romanin (12 Nov, 2020).
"""
function eigs_1site(Hleft,Hloc,Hright,Ainit,x...)

# default parameter
N = 5;
minH = 1e-10;
Aorth = [];

# parsing option
for i in 1:2:length(x)
        if x[i] == "Aorth"
            Aorth = x[i+1];
        elseif x[i] == "N"
            N = x[i+1];
        elseif x[i] == "minH"
            minH = x[i+1];
        else
            error("ERROR: check input!");
    	end
end

# size of ket tensor
Asz = (size(Ainit,1),size(Ainit,2),size(Ainit,3));

# initialize Ainit and Aorth
if !isempty(Aorth)
    Aorth = Aorth/norm(Aorth); # normalize Aorth
    # orthogonalize Ainit w.r.t. Aorth
    Atmp = Aorth[:];
    Atmp2 = Ainit[:];
    Atmp2 = Atmp2 - Atmp*(Atmp'*Atmp2);
    Atmp2 = Atmp2 - Atmp*(Atmp'*Atmp2); # twice, to reduce numerical noise
    Ainit = reshape(Atmp2,Asz); # reshape to rank-3 tensor
end
Ainit = Ainit/norm(Ainit); # normalize Ainit

# Krylov vectors (vectorized tensors)
Akr = zeros(length(Ainit),!isempty(Aorth)+N+1);
if !isempty(Aorth)
    Akr[:,1] = Aorth[:];
end
Akr[:,!isempty(Aorth)+1] = Ainit[:];

# In the Krylov basis, the Hamiltonian becomes tridiagonal
ff = zeros(N,1); # 1st diagonal
gg = zeros(N+1,1); # main diagonal

for itN in 1:(N+1)
    # contract Hamiltonian with ket tensor
    Atmp = eigs_1site_HA(Hleft,Hloc,Hright,reshape(Akr[:,!isempty(Aorth)+itN],Asz));
    Atmp = Atmp[:]; # vectorize

    gg[itN] = Akr[:,!isempty(Aorth)+itN]'*Atmp; # diagonal element, "on-site energy"

    if itN < (N+1)
        # orthogonalize Atmp w.r.t. the previous ket tensors
        Atmp = Atmp - Akr[:,1:(!isempty(Aorth)+itN)]*(Akr[:,1:(!isempty(Aorth)+itN)]'*Atmp);
        Atmp = Atmp - Akr[:,1:(!isempty(Aorth)+itN)]*(Akr[:,1:(!isempty(Aorth)+itN)]'*Atmp); # twice, to reduce numerical noise

        # norm
        ff[itN] = norm(Atmp);

        if ff[itN] > minH
            Akr[:,!isempty(Aorth)+itN+1] = Atmp/ff[itN];
        else
            # stop iteration; truncate ff, gg
	    ff = ff[1:(itN-1)];
	    gg = gg[1:itN];
	    Akr = Akr[:,(1:!isempty(Aorth)+itN)];
            break
        end
    end
end

# remove the column coresponding to Aorth, as it is not needed afterwards
if !isempty(Aorth)
    Akr = Akr[:,2:end];
end

# Hamiltonian in the Krylov basis
Hkr = zeros(length(gg),length(gg));
for i in 1:length(ff)
	Hkr[i,i] = gg[i]/2.0;
	Hkr[i,i+1] = ff[i];
	if i == length(ff)
		Hkr[i+1,i+1] = gg[i+1]/2.0;
	end
end
Hkr = Hkr + Hkr';
Ekr,Vkr = eigen((Hkr+Hkr')/2.0);

# ground state
Aeff = Akr*Vkr[:,1];
Aeff = Aeff/norm(Aeff); # normalize
Aeff = reshape(Aeff,Asz); # reshape to rank-3 tensor

# ground-state energy; measure again
Atmp = eigs_1site_HA(Hleft,Hloc,Hright,Aeff);
Eeff = Aeff[:]'*Atmp[:];

return Aeff, Eeff

end

"""
 < Description >

 Aout = eigs_1site_HA (Hleft,Hloc,Hright,Ain)

 Apply the effective Hamitonian for the site-canonical MPS.
 
 < Input >
 Hleft, Hloc, Hright: [tensors] Refer to the description of the variables
       with the same names, in 'DMRG_1site_eigs'.
 Ain : [tensor] A ket tensor at a lattice site, to be applied by the
       effective Hamiltonian.

 < Output >
 Aout : [tensor] A ket tensor at a lattice site, after the application of
   the effective Hamiltonian to Ain.

 Written by S.Lee (May 23,2020)
 Updated by S.Lee (May 27,2020): Minor change.
 Julia version by D. Romanin (12/11/202)
"""
function eigs_1site_HA(Hleft,Hloc,Hright,Ain)

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

end

